# DeepX 元程实现设计

> DeepX 是元程规范的第一种实现。本文档定义 DeepX 如何用 Redis + op-cuda/metal + heapmem-cuda/metal + VM 进程
> 将元程的抽象模型落地为可运行的分布式计算系统。

## 1. 架构总览

### 1.1 四个核心进程

```
┌─ pysdk (算法前端) ─────────────────────────────────────────────┐
│  deepx/nn/  → 定义模型结构                                      │
│  直接发送 IR 序列到 Redis，或经编译器编译后发送                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │ SET /funcs/..., SET /vthread/...
                       ▼
┌─ Redis (KV 空间) ──────────────────────────────────────────────┐
│  /funcs/   函数定义                                              │
│  /vthread/ 所有线程的执行状态                                     │
│  堆变量    模型权重、训练数据、checkpoint                         │
│  List      命令队列 (VM→op-plat, op-plat→VM)                    │
│  Lock      互斥锁                                                │
└──┬────────────┬──────────────┬──────────────────────────────────┘
   │            │              │
   ▼            ▼              ▼
┌──────┐  ┌──────────┐  ┌───────────┐
│  VM  │  │ op-plat  │  │ heap-plat │
│ 进程 │  │ 进程     │  │ 进程      │
│      │  │          │  │           │
│ 解释 │  │ 执行张量  │  │ 管理tensor│
│ 调度 │  │ 计算      │  │ 生命周期  │
│      │  │ GPU kernel│  │ shm分配   │
└──────┘  └──────────┘  └───────────┘
```

### 1.2 进程的被动性

所有核心进程（VM、op-plat、heap-plat）都**被动**执行，不主动发起操作：

| 进程 | 驱动方式 | 消费来源 |
|------|---------|---------|
| VM | BLPOP 等待通知 | `notify:vm` (新 vthread 创建) + `done:<vtid>` (op-plat 完成) |
| op-plat | RPOP 或 BLPOP | `cmd:op-cuda:<device>` (计算指令) |
| heap-plat | RPOP 或 BLPOP | `cmd:heapmem:<device>` (生命周期指令) |

## 2. Redis Key 路径约定

### 2.1 保留路径

```
/funcs/                 全局函数定义
/vthread/               所有 vthread 执行状态
/sys/                   系统信息
/cmd/                   命令队列前缀
/notify/                通知队列前缀
/done/                  完成通知队列前缀
```

### 2.2 函数路径 (源码层)

`/funcs/` 存储**人类可读**的 dxlang 文本源码。每条指令一个 key，value 是自包含的 dxlang 字符串。

```
/funcs/<name>                      函数签名 (dxlang 文本)
/funcs/<name>/0                    第 0 条指令, 如 "matmul(A, B) -> ./Y"
/funcs/<name>/1                    第 1 条指令, 如 "mul(./Y, alpha) -> ./Y"
/funcs/<name>/1/true/0             分支 true 的第 0 条
/funcs/<name>/1/false/0            分支 false 的第 0 条
/funcs/<name>/2/body/0             循环体的第 0 条
```

注：`/funcs/` 是**源码层**，`/vthread/` 是**执行层**。VM 在 CALL 时将源码翻译为执行层格式（§3.2）。

### 2.3 Vthread 路径 (执行层)

`/vthread/` 存储**机器优化**的执行层格式。指令展开为 `[addr0, addr1]` 二维坐标。
命名槽位 (`./mm`) 与指令坐标是平级子 key，互不嵌套。

```
/vthread/<vtid>                     → {"pc":"[3,0]", "status":"running"}  ← vthread 自身
/vthread/<vtid>/[0,0]              指令 #0 操作码
/vthread/<vtid>/[0,-1]             指令 #0 读参数 #1
/vthread/<vtid>/[0,1]              指令 #0 写参数 #1
/vthread/<vtid>/[0,0]/[0,0]        子栈 (CALL 产生)
/vthread/<vtid>/a                  命名槽位 (与 [0,0] 平级)
/vthread/<vtid>/mm                 局部变量 (./mm 解析结果)
```

注：`pc` 和 `status` 是 `/vthread/<vtid>` 的 value 中的字段，不是独立的子 key。

### 2.4 命令队列

```
cmd:op-cuda:<device_id>            op-cuda 的命令队列 (如 cmd:op-cuda:0)
cmd:op-metal:<device_id>           op-metal 的命令队列
cmd:heapmem:<device_id>            heap-plat 的命令队列
done:<vtid>                        vthread 的完成通知队列
notify:vm                          VM 的唤醒通知队列
```

### 2.5 系统路径

```
/sys/op-plat/<id>                  op-plat 注册信息 {type, device, status, load}
/sys/heap-plat/<id>                heap-plat 注册信息
/sys/vm/<id>                       VM 实例注册信息
/sys/config                        DeepX 全局配置
```

### 2.6 堆变量（隐式命名空间）

除上述保留路径外，所有其他路径均为堆变量。命名规范由上层决定：

```
/models/bert/encoder/0/weights     ← 建议: 模型名/层名/参数名
/data/cifar10/train                ← 建议: 数据集/用途
/checkpoints/run_001/step_1000     ← 建议: 运行ID/步数
/rl/state_buffer                   ← 自由命名
```

## 3. VM 进程设计

### 3.1 VM 状态机

```
         ┌─────────┐
         │  start  │
         └────┬────┘
              │ 扫描 /vthread/ 找到 status=init 的 vthread
         ┌────▼────┐
    ┌───►│  idle   │◄──────────────┐
    │    └────┬────┘               │
    │         │ 拾取 vthread       │
    │    ┌────▼────┐               │
    │    │ running │               │
    │    └────┬────┘               │
    │         │                    │
    │    ┌────┴────┐               │
    │    │         │               │
    │ 计算指令  控制流/生命周期      │
    │    │         │               │
    │    ▼         ▼               │
    │  PUSH      VM 直接处理        │
    │  到op-plat  (call/if/        │
    │    │       return/del)       │
    │    ▼         │               │
    │  BLPOP       │               │
    │  done:<vtid> │               │
    │    │         │               │
    │    ▼         │               │
    │  PC++        │               │
    │    │         │               │
    └────┴─────────┘               │
         │                         │
         │ vthread 全部执行完       │
    ┌────▼────┐                    │
    │  wait   │────────────────────┘
    │ (BLPOP  │  新 vthread 创建
    │ notify) │
    └─────────┘
```

### 3.2 VM 执行循环

```
1. 扫描 /vthread/ 子树，找到 status=init 的 vthread
   → 有: 拾取执行
   → 无: BLPOP notify:vm (等待新 vthread 创建)

2. 对当前 vthread:
   a) GET /vthread/<vtid> → {pc: "[n,0]", status: "running", ...}
   b) GET /vthread/<vtid>/[n,0] → opcode

3. 指令分发:
   
   [张量计算指令] add, matmul, relu, mul, exp, ...
     → 构造 op-plat 任务包 (§4.2)
     → PUSH cmd:op-cuda:<device>
     → SET /vthread/<vtid> = {pc: "[n,0]", status: "wait", ...}
     → BLPOP done:<vtid>
     → 醒来后检查结果
     → SET /vthread/<vtid> = {pc: "[n+1,0]", status: "running", ...}
   
   [控制流指令]
     call (VM 翻译: 源码层→执行层, eager):
       → 读取 [n,-1] func_name, [n,-2..] args, [n,1] return_slot
       → 批读取 /funcs/<func_name>/ 下所有指令 (一次 Redis MGET/Pipeline)
       → 逐条翻译: 解析 dxlang → 形参替换为实参 → 展开为 [i,0],[i,-1]...[i,1]
       → 批写入 /vthread/<vtid>/[n,0]/ 子栈
       → SET /vthread/<vtid> = {pc: "[n,0]/[0,0]", status: "running", ...}  (进入子栈)
     
     return:
       → 将返回值写入父栈 CALL 指令的写参数槽位
       → DELETE 当前子栈 KV 路径
       → SET /vthread/<vtid> = {pc: "[n+1,0]", status: "running", ...}  (父栈下一条)
    
     if:
       → 读取条件: [n,-1] = cond, [n,0]/true/0 和 [n,0]/false/0
       → 根据 cond 结果设置 PC
    
     for:
       → 初始化迭代器, 循环 body 直到耗尽
   
   [生命周期指令]
     newtensor:
       → PUSH cmd:heapmem:<device>
       → 等待完成 (或同步)
     
     deltensor:
       → PUSH cmd:heapmem:<device>
     
     var:
       → SET /vthread/<vtid>/<name> = <value>  (基础类型直存 Redis)

4. vthread 执行完毕 (pc 超出 seq 范围):
   → SET /vthread/<vtid> = {pc: "...", status: "done", ...}
   → 清理 /vthread/<vtid>/ 子树 (GC)
   → 回到步骤 1
```

### 3.3 多 VM 并行

多个 VM 进程可并行运行，每个 VM 独立拾取 status=init 的 vthread：

```
VM-1 拾取 /vthread/1/ → 执行
VM-2 拾取 /vthread/2/ → 执行
VM-3 空闲 → BLPOP notify:vm

避免竞争: 使用 Redis 原子操作标记 vthread 已被拾取
  WATCH /vthread/<vtid>
  GET → {pc: "...", status: "init"}
  MULTI
  SET /vthread/<vtid> = {pc: "...", status: "running"}
  EXEC
  → 只有一个 VM 能成功
```

TODO: 多 VM 间的负载均衡策略，待开发时确定。

### 3.4 性能关键：批读取与本地缓存

VM 每次从 Redis GET 一个 key 都有网络往返延迟 (~100μs)。
相比之下，VM 读自己进程内存仅 ~ns 级。减少 Redis 访问是性能关键。

**当前设计的 Redis 访问次数（逐条执行 20 条指令的 func）：**

```
每步操作                        Redis 访问次数
──────────────────────────     ──────────────
CALL 翻译: 读取 func 源码          1 次 (MGET 批量取所有指令)
CALL 翻译: 写入子栈               1 次 (Pipeline 批量写)
逐条执行: GET pc + GET opcode    20 × 2 = 40 次
发射到 op-plat: PUSH             20 次
等待完成: BLPOP                  20 次
更新 PC: SET                     20 次
总计                             ~102 次 Redis 往返
```

**优化方向：**

**(a) 批读取 — 减少执行循环的 Redis 次数：**

```
当前 (逐条):
  GET /vthread/1 → pc=[0,0]     ← 1 次
  GET /vthread/1/[0,0] → opcode      ← 1 次
  SET /vthread/1 = {pc:[0,0], status:"wait"}
  ...等 op-plat...
  GET /vthread/1 → pc=[1,0]     ← 1 次
  GET /vthread/1/[1,0] → opcode      ← 1 次

优化后 (VM 本地预取一批指令):
  MGET /vthread/1, /vthread/1/[0,0], /vthread/1/[0,-1], /vthread/1/[0,1],
       /vthread/1/[1,0], /vthread/1/[1,-1], /vthread/1/[1,1], ...
       /vthread/1/[19,0], /vthread/1/[19,-1], /vthread/1/[19,1]
  → 1 次往返，拿到整个子栈的所有指令，缓存到 VM 本地内存
  → 后续逐条执行时从本地缓存读取，零 Redis 访问
  → 每批只改 PC 和 status (SET /vthread/<vtid>)
```

**(b) MGET 批量取 tensor 元信息：**

```
发射指令前，需要读取参数指向的 tensor 元信息 (dtype, shape, shm):
  GET /vthread/1/a   ← 1 次
  GET /vthread/1/b   ← 1 次
  GET /models/W      ← 1 次
  
  → MGET /vthread/1/a, /vthread/1/b, /models/W  ← 1 次
```

**(c) 算子融合 — 编译器在 /funcs/ 层等价替换：**

算子融合是编译器的职责，发生在**源码层**。编译器分析 `/funcs/` 下的函数体，
将可融合的连续算子替换为等价的一条融合指令，重写 `/funcs/`。VM 和 op-plat 对融合无感知。

```
融合前 (/funcs/forward 的原始代码):
  /funcs/forward/0 = matmul(A, B) -> ./mm
  /funcs/forward/1 = add(./mm, b) -> ./mm
  /funcs/forward/2 = relu(./mm) -> ./out

编译器分析 → 融合替换 → 重写 /funcs/:

融合后:
  /funcs/forward/0 = fused_matmul_add_relu(A, B, b) -> ./out
  (原来 3 条指令 → 1 条)
```

VM 执行时的效果（透明受益）：

```
融合前: VM PUSH matmul → 等 → PUSH add → 等 → PUSH relu → 等
        → 3 次 VM↔Redis 往返 + 3 次 VM↔op-plat 往返

融合后: VM PUSH fused_matmul_add_relu → 等
        → 1 次调度, 减少 2/3 的 VM 和 Redis 开销
        → GPU 侧也减少 kernel launch 次数 (3→1)
```

TODO: 编译器融合规则的实现，待编译器设计阶段确定。

**(d) VM 本地子栈缓存：**

```
VM 在执行一个 vthread 的某个栈帧时:
  1. CALL 翻译完成后，整个子栈的 [i,j] key 一次性 MGET 到本地 map
  2. 执行期间，opcode 和参数查找全走本地内存 (O(1) hash)
  3. 仅以下操作写 Redis:
     - 更新 PC (SET /vthread/<vtid>)
     - PUSH 到 op-plat 命令队列
     - BLPOP 完成通知
  4. RETURN 时 DELETE 子栈 (1 次批量操作)
```

TODO: 本地缓存的失效策略（当外部 WATCH 修改了 vthread 状态时），待开发时确定。

## 4. op-plat 协议

### 4.1 op-plat 生命周期

```
1. 启动 → 注册到 /sys/op-plat/<id>:
   { "type": "op-cuda", "device": "gpu0", "status": "idle",
     "capabilities": ["add", "matmul", "relu", "softmax", ...] }

2. 进入消费循环:
   while true:
     RPOP cmd:op-cuda:0  (或 BLPOP 阻塞等待)
     解析指令
     执行 GPU kernel
     LPUSH done:<vtid> 完成通知

3. 退出 → DELETE /sys/op-plat/<id>
```

### 4.2 指令格式

VM PUSH 到 `cmd:op-cuda:<device>` 的指令：

```json
{
  "vtid": "1",
  "pc": "[3,0]",
  "opcode": "matmul",
  "inputs": [
    {
      "key": "/vthread/1/a",
      "dtype": "f32",
      "shape": [1024, 512],
      "address": {
        "node": "n1",
        "device": "gpu0",
        "type": "shm",
        "shm_name": "/deepx_t_abc123",
        "byte_size": 2097152
      }
    }
  ],
  "outputs": [
    {
      "key": "/vthread/1/c",
      "dtype": "f32",
      "shape": [1024, 256],
      "address": {
        "node": "n1",
        "device": "gpu0",
        "type": "shm",
        "shm_name": "/deepx_t_def456",
        "byte_size": 1048576
      }
    }
  ],
  "params": {
    "transpose_a": false,
    "transpose_b": true
  }
}
```

TODO: 是否直接发送 GPU 指针而非 shm_name，待开发时根据性能需求确定。

### 4.3 完成通知格式

op-plat 计算完成后 LPUSH 到 `done:<vtid>`：

```json
{
  "pc": "[3,0]",
  "status": "ok",
  "outputs_updated": [
    {"key": "/vthread/1/c", "new_shape": [1024, 256]}
  ]
}
```

错误情况：
```json
{
  "pc": "[3,0]",
  "status": "error",
  "error": {
    "code": "GPU_OOM",
    "message": "out of memory: requested 2GB, available 1.5GB"
  }
}
```

### 4.4 批量发射

VM 可将无依赖的多条指令打包为一批，一次 PUSH：

```json
{
  "batch": [
    {"pc": "[3,0]", "opcode": "add", ...},
    {"pc": "[4,0]", "opcode": "mul", ...},
    {"pc": "[5,0]", "opcode": "relu", ...}
  ]
}
```

op-plat 可以并行执行 batch 内的指令（不同 CUDA stream 或 Metal command queue）。

TODO: 批量发射的依赖分析由编译器完成还是 VM 运行时分析，待开发时确定。当前阶段 VM 逐条发送。

## 5. heap-plat 协议

### 5.1 heap-plat 生命周期

```
1. 启动 → 注册到 /sys/heap-plat/<id>:
   { "type": "heapmem-cuda", "device": "gpu0", "status": "idle" }

2. 进入消费循环:
   while true:
     RPOP cmd:heapmem:0  (或 BLPOP)
     解析指令
     执行 shm 分配/释放/克隆
     回复到 done:<vtid> 或直接 SET 元信息到堆路径

3. 退出 → DELETE /sys/heap-plat/<id>
```

### 5.2 指令格式

**创建 tensor:**
```json
{
  "vtid": "1",
  "pc": "[0,0]",
  "op": "newtensor",
  "key": "/models/weights",
  "dtype": "f32",
  "shape": [1024, 512],
  "device": "gpu0"
}
```

heap-plat 执行：
1. 分配 shm + GPU buffer
2. SET `/models/weights` = 完整元信息（含 shm_name、byte_size、device）

**删除 tensor:**
```json
{
  "vtid": "1",
  "pc": "[5,0]",
  "op": "deltensor",
  "key": "/models/weights"
}
```

**克隆 tensor:**
```json
{
  "vtid": "1",
  "pc": "[0,0]",
  "op": "clonetensor",
  "src": "/models/weights",
  "dst": "/models/weights_gpu1",
  "device": "gpu1"
}
```

TODO: 引用计数机制 (refcount) 是否在 heap-plat 侧实现，还是由 VM 统一管理，待开发时确定。

## 6. Tensor 元信息格式

堆变量和 vthread 命名槽位的 value 的 tensor 元信息：

```json
{
  "dtype": "f32",
  "shape": [1024, 512],
  "byte_size": 2097152,
  "device": "gpu0",
  "address": {
    "node": "n1",
    "type": "shm",
    "shm_name": "/deepx_t_abc123"
  },
  "ctime": 1714000000,
  "version": 5
}
```

对于 vthread 命名槽位中的基础类型，value 直接是字面量：
```
/vthread/1/a = 1          (int)
/vthread/1/b = 3.14       (float)
/vthread/1/flag = true    (bool)
```

## 7. pysdk 接口设计

### 7.1 当前模式：直接发送 IR 序列

pysdk 直接将 deepxIR 序列写入 `/funcs/` 和 `/vthread/`：

```python
# front/py/deepx/nn/functional/ 下的代码模式 (当前)
kv = KVSpace()

# 1. 定义 func
kv.set("/funcs/forward", "(forward(A, B) -> (C))")
kv.set("/funcs/forward/0", "matmul(A, B) -> ./mm")
kv.set("/funcs/forward/1", "relu(./mm) -> C")

# 2. 创建 vthread
vtid = kv.alloc_vtid()
kv.set(f"/vthread/{vtid}", {"pc": "[0,0]", "status": "init"})

# 3. 写入入口指令 (call main)
kv.set(f"/vthread/{vtid}/[0,0]", "call")
kv.set(f"/vthread/{vtid}/[0,-1]", "forward")
kv.set(f"/vthread/{vtid}/[0,-2]", "/models/A")
kv.set(f"/vthread/{vtid}/[0,-3]", "/models/B")
kv.set(f"/vthread/{vtid}/[0,1]", "./C")

# 4. 通知 VM
kv.push("notify:vm", {"event": "new_vthread", "vtid": vtid})
```

### 7.2 未来模式：经编译器

```
pysdk 发送全局 func 定义 → 编译器 (dxlang compiler)
  → 依赖分析 → 插入 deltensor → 算子融合 → 数据并行标注
  → 输出优化后的 /funcs/
  → 创建 /vthread/
```

TODO: 编译器的输入格式 (Python AST? dxlang DSL? deepxIR?) 和输出约定，待设计阶段确定。

### 7.3 Func 缓存

pysdk 维护本地 func 缓存，避免重复发送：

```python
class FuncCache:
    def set_func(self, name, signature, instructions):
        if self._cache.get(name) == hash(instructions):
            return  # 未变化，跳过
        kv.set(f"/funcs/{name}", signature)
        for i, inst in enumerate(instructions):
            kv.set(f"/funcs/{name}/{i}", inst)
        self._cache[name] = hash(instructions)
```

## 8. 启动流程

### 8.1 集群启动顺序

```
1. Redis 启动 (已有)
2. heap-plat 启动 × N (每个 GPU 一个)
   → 注册到 /sys/heap-plat/<id>
3. op-plat 启动 × N (每个 GPU 一个)
   → 注册到 /sys/op-plat/<id>
   → 开始 BLPOP cmd:op-cuda:<device>
4. VM 启动 × M (可多个)
   → 注册到 /sys/vm/<id>
   → 扫描 /vthread/ 或 BLPOP notify:vm
5. pysdk 启动
   → 发送 func 定义
   → 创建 vthread
   → PUSH notify:vm
```

### 8.2 Vthread 创建流程

```
pysdk:
  1. 分配 vtid (Redis INCR /sys/vtid_counter)
  2. SET /vthread/<vtid> = {"pc": "[0,0]", "status": "init"}
  3. 写入入口指令序列
  4. PUSH notify:vm {"event": "new_vthread", "vtid": "<vtid>"}

VM:
  1. BLPOP notify:vm → 收到事件
  2. WATCH /vthread/<vtid>
  3. GET → {pc: "...", status: "init"}
  4. MULTI → SET /vthread/<vtid> = {pc: "...", status: "running"} → EXEC
  5. 开始执行循环 (§3.2)
```

## 9. 错误处理

### 9.1 错误分类

| 错误类型 | 示例 | 处理方式 |
|---------|------|---------|
| op-plat 执行失败 | GPU OOM, 数值溢出 | status→error, error 信息写入 /vthread/<vtid> 的 value |
| heap-plat 执行失败 | shm 分配失败, 磁盘满 | status→error |
| 超时 | op-plat 无响应 | VM 超时后 status→error |
| 锁冲突 | LOCK 超时 | 调用者决定重试或报错 |

### 9.2 错误传播

```
op-plat 返回 error:
  1. VM BLPOP 收到 error 完成通知
  2. SET /vthread/<vtid> = {pc: "[n,0]", status: "error", error: {...}}
  3. VM 释放该 vthread, 回到 idle 状态

pysdk 可通过 GET /vthread/<vtid> 检查 status 字段感知错误
TODO: 错误恢复策略 (重试 / 跳过 / 降级) 待开发时确定
```

## 10. 监控与运维

### 10.1 系统状态查询

```
GET /sys/op-plat/0     → op-plat 状态和负载
GET /sys/heap-plat/0   → heap-plat 状态
KEYS /vthread/*        → 所有 vthread 列表
GET /vthread/1         → 特定 vthread 的状态 (含 pc 和 status 字段)
```

### 10.2 liveness 检测

TODO: 各进程的心跳机制和故障检测，待开发时确定。

## 11. 目录结构映射

```
executor/
├── vm/                    VM 进程 (新增)
│   └── src/main.mm
├── op-metal/              op-plat (Metal 实现, 已有)
├── op-cuda/               op-plat (CUDA 实现, 待开发)
├── heapmem-metal/         heap-plat (Metal 实现, 已有)
├── heapmem-cuda/          heap-plat (CUDA 实现, 待开发)
└── common-metal/          共享库 (已有)

front/
└── py/deepx/              pysdk
    └── nn/functional/     deepxIR 算子序列发送

doc/
└── metaproc/              元程设计文档
    ├── spec-v1.md         元程规范 v1 (抽象模型)
    ├── deepx-design.md    本文件 (DeepX 实现设计)
    └── CONVERSATION.md    设计对话记录
```

## 12. 待开发时确定的问题

| 问题 | 当前状态 |
|------|---------|
| 批量发射的依赖分析 (编译器 vs VM 运行时) | 暂定 VM 逐条发送 |
| 指令中发送物理地址 vs key 引用 | 暂定发送完整 tensor 元信息 |
| 多 VM 负载均衡策略 | 暂定 SETNX 竞争拾取 |
| 引用计数 (heap-plat vs VM) | 暂未实现 |
| 编译器输入格式与输出约定 | pysdk 直接发送 IR 可工作，编译器为后续优化 |
| 多卡/跨节点 tensor 迁移 | 暂不实现，当单机处理 |
| 动态图支持 | 暂不支持 |
| 进程心跳与故障检测 | 暂不实现 |
| 错误恢复策略 | 暂定 status→error，等待人工介入 |
