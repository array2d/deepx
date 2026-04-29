# 元程规范 (Metaproc Specification) v1

> 元程是一种基于 KV 寻址空间的分布式计算模型。
> 本规范定义元程的抽象语义，不依赖任何具体实现（Redis、CUDA、Metal 等）。
> 实现方只需提供一个满足 §2 要求的 KV 空间，即可运行元程程序。

元程系统由 5 个核心组成：

| 核心 | 角色 | 说明 |
|------|------|------|
| KV 空间 | 全局状态存储 | 命名空间、命令队列、锁 |
| 算法前端 | 源码注册 | 将 func 定义写入 KV 空间，创建执行单元 |
| op-plat | 计算 | 被动消费指令，执行张量运算 |
| heap-plat | 堆管理 | tensor 对象生命周期：创建/删除/克隆 |
| VM | 解释执行 | CALL 翻译、指令路由、状态推进 |

---

# 第一部分：元程模型

## 1. 核心概念

### 1.1 元程 (Metaproc)

一个 **KV 空间** 就是一个元程。

类比：一个虚拟地址空间就是一个 OS 进程。
KV 空间内的所有路径属于同一个元程，所有 vthread 共享该空间内的堆数据。

### 1.2 元线程 (Vthread)

元程内的执行流。一个元程可以有一个或多个 vthread 并行执行。
vthread 拥有私有的调用栈，但共享元程的堆数据。

类比：OS 进程内的线程。线程私有栈，共享堆。

### 1.3 函数 (Func)

定义在 `/src/func/` 下的可复用代码单元。编译器优化后存于 `/op/<backend>/func/`。
所有 vthread 可以 CALL 同一个 func。

类比：共享库中的函数，或 ELF 的 .text 段中的代码。

### 1.4 堆 (Heap)

元程内全局共享的长生命周期数据。所有非保留路径默认即堆。
堆数据的 value 包含 dtype、shape、物理地址等元信息。

类比：OS 进程的 .data / .bss 段 + mmap 区域。

### 1.5 栈 (Stack)

每个 vthread 私有的执行状态。包括指令序列、局部变量、PC。
栈在 CALL 时扩展（创建子维度），RETURN 时收缩（删除子维度）。

类比：OS 线程的调用栈 (call stack)。

## 2. KV 空间要求

元程要求 KV 空间提供以下能力。实现方可使用任何满足这些要求的存储系统。

### 2.1 基本操作

| 操作 | 语义 |
|------|------|
| `GET(key)` | 读取 key 对应的 value |
| `SET(key, value)` | 写入 value，覆盖或新建 |
| `DELETE(key)` | 删除 key 及其所有子 key（递归） |
| `EXISTS(key)` | 判断 key 是否存在 |

### 2.2 通知队列

KV 空间必须提供类似消息队列的通知机制：

| 操作 | 语义 |
|------|------|
| `PUSH(list_key, value)` | 向队列尾部追加一条消息 |
| `POP(list_key, timeout)` | 阻塞地从队列头部取出一条消息。超时返回空 |
| `LEN(list_key)` | 返回队列当前长度 |

队列是 FIFO 的。支持多个消费者竞争 POP（每条消息仅被一个消费者获取）。

### 2.3 锁

| 操作 | 语义 |
|------|------|
| `LOCK(key, holder, ttl)` | 尝试获取排他锁。若已被持有则失败。支持 TTL 自动释放 |
| `UNLOCK(key, holder)` | 释放锁 |

### 2.4 值类型

KV 空间必须能直接存储以下基础类型（无需外部存储）：

- 整数 (int32, int64)
- 浮点数 (float32, float64)
- 布尔值
- 短字符串 (建议 < 1KB)
- 结构化数据 (JSON 或等价格式)

对于超大二进制数据（如 tensor 的实际数据），KV 空间仅存**引用**（物理地址），数据本身存于外部存储系统。

## 3. 路径空间约定

### 3.1 保留路径

以下路径前缀由元程运行时保留：

| 路径 | 用途 |
|------|------|
| `/src/func/` | 函数源码 (平台无关的 dxlang 文本) |
| `/op/<backend>/func/` | 后端编译产物 (融合、拆分、设备标注后的 func) |
| `/vthread/` | 所有 vthread 的执行状态 |
| `/sys/` | 系统信息（op-plat、heap-plat 注册等） |

### 3.2 堆的隐式命名空间

除保留路径外，KV 空间中的所有其他路径均为堆路径。
堆路径的具体命名空间划分由上层（如 pysdk）自行管理。

```
例:
  /models/bert/weights          ← 堆
  /data/cifar10/train           ← 堆
  /checkpoints/step1000         ← 堆
  /src/func/forward             ← 函数源码 (保留)
  /op/op-cuda/func/forward      ← 编译产物 (保留)
  /vthread/1/...                ← 栈 (保留)
```

### 3.3 相对路径

vthread 栈内使用 `./` 前缀表示相对路径，解析为 `/vthread/<vtid>/` 下的平级命名槽位。

```
./mm       → /vthread/<vtid>/mm       (命名槽位, 与 [0,0] 等指令坐标平级)
./bias     → /vthread/<vtid>/bias
../a       → /vthread/<vtid>/../a     (不推荐，但技术上可行)
```

命名槽位和指令坐标 `[addr0, addr1]` 是 `/vthread/<vtid>/` 下的**平级子 key**：
- 命名槽位：人读的命名空间，存放 tensor 元信息
- 指令坐标：机器执行的指令序列，VM 据此推进 PC

## 4. 函数定义

### 4.1 三层架构

函数有三种表示形式，分别服务于不同的角色：

| 层 | 位置 | 角色 | 格式 |
|----|------|------|------|
| 源码层 | `/src/func/<name>` | pysdk 写入，人类可读 | dxlang 文本 |
| 编译层 | `/op/<backend>/func/<name>` | 编译器产出，VM 读取 | 后端优化后的 dxlang 文本 |
| 执行层 | `/vthread/<vtid>/` | VM 翻译后，机器执行 | `[addr0, addr1]` 二维坐标 |

```
数据流:
  pysdk → /src/func/forward           (源码)
              │
              ▼ 编译器 (融合、拆分、设备标注)
  /op/op-cuda/func/forward            (CUDA 编译产物)
  /op/op-metal/func/forward           (Metal 编译产物, 可能不同)
              │
              ▼ VM CALL 时读取 + eager 翻译
  /vthread/<vtid>/[n,0]/[0,0]...     (执行层)
```

### 4.2 路径结构

```
/src/func/<func_name>               → 函数签名 (dxlang 文本)
/src/func/<func_name>/0             → 第 0 条指令
/src/func/<func_name>/1             → 第 1 条指令
...

/op/op-cuda/func/<func_name>        → 编译后的函数签名
/op/op-cuda/func/<func_name>/0      → 第 0 条指令 (可能已融合/拆分)
...
```

### 4.3 函数签名

`/src/func/<func_name>` 的值定义函数的类型签名：

```
(func_name((ro_p1:type1, ro_p2:type2, ...) -> (w_p1:type3, w_p2:type4, ...))
```

- 左侧 `()` 内为只读参数
- 右侧 `()` 内为写入参数（返回值）

### 4.4 指令

`/src/func/<func_name>/N` 的值是一条指令。

**指令格式（左读右写）：**

```
opcode(read_param_1, read_param_2, ...) -> write_param_1, write_param_2
```

- `->` 左侧为读取的输入
- `->` 右侧为写入的输出
- 参数可以是 `./` 相对路径（局部变量）、绝对堆路径、或立即数

**控制流指令：**

```
if(cond) ->                    ← 分支入口
  /src/func/<name>/N/true/0   ← true 分支第一条指令
  /src/func/<name>/N/false/0  ← false 分支第一条指令

for(iterator) ->               ← 循环入口
  /src/func/<name>/N/body/0   ← 循环体第一条指令
```

### 4.5 编译层

编译器读取 `/src/func/` 的源码，产出后端专属的编译产物到 `/op/<backend>/func/`：

```
/src/func/gemm/0 = matmul(A, B) -> ./Y
/src/func/gemm/1 = mul(./Y, alpha) -> ./Y
/src/func/gemm/2 = mul(C, beta) -> ./C
/src/func/gemm/3 = add(./Y, ./C) -> ./Y

     ↓ 编译器: 算子融合

/op/op-cuda/func/gemm/0 = fused_matmul_mul_mul_add(A, B, alpha, C, beta) -> ./Y
/op/op-metal/func/gemm/0 = matmul(A, B) -> ./tmp1
/op/op-metal/func/gemm/1 = mul(./tmp1, alpha) -> ./tmp2
/op/op-metal/func/gemm/2 = mul(C, beta) -> ./tmp3
/op/op-metal/func/gemm/3 = add(./tmp2, ./tmp3) -> ./Y
```

不同后端的编译产物可以不同（CUDA 融合了 4→1，Metal 保持 4 条）。
VM 在 CALL 时读取对应后端的 `/op/<backend>/func/`，而非 `/src/func/`。

### 4.6 示例

```
/src/func/gemm = (gemm(A:tensor<f32>, B:tensor<f32>, alpha:f32, beta:f32, C:tensor<f32>) -> (Y:tensor<f32>))

/src/func/gemm/0 = matmul(A, B) -> ./Y
/src/func/gemm/1 = mul(./Y, alpha) -> ./Y
/src/func/gemm/2 = mul(C, beta) -> ./C
/src/func/gemm/3 = add(./Y, ./C) -> ./Y
```

### 4.7 源码层与执行层

源码层（`/src/func/`、`/op/<backend>/func/`）与执行层（`/vthread/`）的对比：

| | 源码层 / 编译层 | 执行层 |
|---|---|---|
| 格式 | dxlang 文本：`matmul(A,B)->./Y` | 二维坐标：`[0,0]=matmul`, `[0,-1]=/models/A` |
| 目标 | 人读 / 编译器读写 | 机器高效寻址、零解析 |
| 翻译 | — | VM CALL 时 eager 翻译（§6.2） |

### 4.8 op-plat 算子注册

op-plat **程序**（如 op-cuda）定义了一套算子能力。该程序的多个**进程实例**
（如 GPU0 上的 op-cuda、GPU1 上的 op-cuda）共享同一套算子注册。

**程序级路径（静态能力，所有实例共享）：**

```
/op/<program>/list           → ["matmul", "add", "relu", "fused_matmul_add_relu", ...]
/op/<program>/<opname>       → 算子元数据
/op/<program>/func/          → 该程序专属的编译产物
```

**进程级路径（动态状态，每个实例独立）：**

```
/sys/op-plat/<instance>      → {pid, device, status, load, started_at}
```

例：一台机器有 2 张 GPU，运行 2 个 op-cuda 实例：

```
程序级 (共享):
  /op/op-cuda/list = ["matmul", "add", "relu", "fused_matmul_add_relu"]
  /op/op-cuda/matmul = {"category":"matmul", "dtype":["f32","f16"], ...}
  /op/op-cuda/func/forward/0 = fused_matmul_add_relu(A,B,b)->./out

进程级 (独立):
  /sys/op-plat/cuda:0 = {"device":"gpu0", "status":"running", "load":0.3}
  /sys/op-plat/cuda:1 = {"device":"gpu1", "status":"running", "load":0.7}
```

**算子元数据：**

```
/op/op-cuda/matmul = {
    "category": "matmul",
    "dtype": ["f32", "f16", "bf16"],
    "max_shape": [8192, 8192, 8192],
    "fusion_group": "linear"
}

/op/op-cuda/fused_matmul_add_relu = {
    "category": "fused",
    "dtype": ["f32", "f16"],
    "replaces": ["matmul", "add", "relu"]
}
```

**编译器使用注册信息：**

```
融合决策:
  1. GET /op/op-cuda/list → 过滤 category="fused" 的算子
  2. 对每个 fused 算子，读取 replaces 列表
  3. 扫描 /src/func/ 指令序列，滑动窗口匹配 replaces 模式
  4. 匹配成功 → 等价替换 → 写入 /op/op-cuda/func/<name>

拆分决策:
  1. GET /op/op-cuda/<op>/max_shape → 单卡能力上限
  2. 对比 tensor 实际 shape → 超出则拆分为多个子算子
  3. 子算子标注目标设备 → 写入 /op/op-cuda/func/<name>
```

**VM 使用注册信息：**

```
指令路由:
  GET /op/op-cuda/list → 含 "matmul"
  GET /op/op-metal/list → 不含 "matmul"
  → VM 将 matmul 路由到 op-cuda 的某个空闲实例
  → 实例选择基于 /sys/op-plat/cuda:* 的 load 信息
```

## 5. Vthread 执行模型

### 5.1 Vthread 路径结构

```
/vthread/<vtid>              → {"pc":"[2,0]", "status":"running"}   ← vthread 自身 (含 PC 和状态)
/vthread/<vtid>/[0,0]        ← 指令 #0 的操作码
/vthread/<vtid>/[0,-1]       ← 指令 #0 的读参数 #1
/vthread/<vtid>/[0,-2]       ← 指令 #0 的读参数 #2
/vthread/<vtid>/[0, 1]       ← 指令 #0 的写参数 #1
/vthread/<vtid>/[1,0]        ← 指令 #1 的操作码
...
/vthread/<vtid>/a            ← 命名槽位 a (与 [0,0] 平级，非嵌套)
/vthread/<vtid>/b            ← 命名槽位 b
/vthread/<vtid>/mm           ← 局部变量 mm (./mm 解析结果)
/vthread/<vtid>/[2,0]/[0,0]  ← 子栈: 指令 #2 是 CALL，其子栈的指令 #0
```

命名槽位 (`/vthread/<vtid>/a`) 与指令坐标 (`/vthread/<vtid>/[0,0]`) 是 `/vthread/<vtid>/` 下的**平级子 key**。
它们互不嵌套：命名槽位供 tensor 读写，指令坐标供 VM 推进 PC。

### 5.2 二维寻址

vthread 的指令序列使用二维坐标 `[addr0, addr1]` 寻址：

| addr1 | 含义 | 示例 |
|-------|------|------|
| `0` | 操作码 | `call`, `add`, `matmul`, `newtensor` |
| `-1, -2, ...` | 读取参数 (左值) | func 名、输入 tensor、立即数 |
| `1, 2, ...` | 写入参数 (右值) | 输出 tensor、返回值 |

addr0 是序列维，表示指令在栈帧内的顺序位置。

### 5.3 命名槽位

命名槽位是 `/vthread/<vtid>/` 下的平级子 key（如 `/vthread/<vtid>/a`），
与指令坐标 `[addr0, addr1]` 互不嵌套，用于存放局部变量的值。

基础类型（int, float, bool, 短 string）的值直接存储在槽位中。
Tensor 类型的值存储 tensor 元信息（dtype, shape, 物理地址）。

命名槽位在 `/vthread/` 下的路径名与 dxlang 源码中的变量名一致，保证可调试性：
```
源码: matmul(A, B) -> ./mm
执行: /vthread/1/[0,0] = matmul, /vthread/1/[0,-1] = /models/A, ...
      /vthread/1/mm = {dtype:"f32", shape:[1024,256], address:{...}}
```

### 5.4 程序计数器 (PC)

PC 是 `/vthread/<vtid>` 的 value 中的 `pc` 字段，指向当前待执行指令的坐标。

```
/vthread/1 = {"pc": "[3,0]", "status": "running"}
```

`pc` 的值如 `[3,0]` 表示当前位于栈帧指令序列的第 3 条指令。

### 5.5 Vthread 状态

Vthread 状态是 `/vthread/<vtid>` 的 value 中的 `status` 字段：

```
/vthread/1 = {"pc": "[3,0]", "status": "wait"}
```

| 状态 | 含义 |
|------|------|
| `init` | 已创建，待 VM 拾取 |
| `running` | VM 正在调度执行 |
| `wait` | 等待异步操作（op-plat 完成 / 锁释放） |
| `error` | 执行出错 |
| `done` | 执行完毕，栈待 GC |

## 6. CALL 与子栈

### 6.1 CALL 指令

当 vthread 执行到 `call` 指令时：

```
/vthread/<vtid>/[n, 0]  = call
/vthread/<vtid>/[n,-1]  = <func_name>      ← 被调用的函数名
/vthread/<vtid>/[n,-2]  = <arg1>           ← 只读参数
/vthread/<vtid>/[n,-3]  = <arg2>
...
/vthread/<vtid>/[n, 1]  = <ret1>           ← 返回值绑定的槽位
```

### 6.2 子栈创建 — VM 翻译 (eager)

VM 在 CALL 时**一次性**将 `/op/<backend>/func/<func_name>/` 的编译层指令翻译为执行层格式，
复制到 vthread 的子栈。

**翻译是 eager 的**（而非 lazy）：所有指令在 CALL 时完成翻译，执行时零解析开销。

VM 读取的是编译层（`/op/<backend>/func/`），而非源码层（`/src/func/`）。
编译层已经是编译器优化后的产物（融合、拆分、设备标注已完成），
VM 只需做形参替换和坐标展开。

```
翻译前 (/op/op-cuda/func/ 编译层):      翻译后 (/vthread/ 执行层, 子栈):
────────────────────────────────        ──────────────────────────────
/op/op-cuda/func/gemm = (签名)           /vthread/1/[n,0]/  ← 子栈根

/op/op-cuda/func/gemm/0                  /vthread/1/[n,0]/[0,0] = matmul
  = matmul(A, B) -> ./Y                  /vthread/1/[n,0]/[0,-1] = /models/A
                                          /vthread/1/[n,0]/[0,-2] = /models/B
                                          /vthread/1/[n,0]/[0, 1] = ./Y

/op/op-cuda/func/gemm/1                  /vthread/1/[n,0]/[1,0] = mul
  = mul(./Y, alpha) -> ./Y               /vthread/1/[n,0]/[1,-1] = ./Y
                                          /vthread/1/[n,0]/[1,-2] = 1.0
                                          /vthread/1/[n,0]/[1, 1] = ./Y
```

**翻译步骤：**

```
1. 从 CALL 指令的读参数中获取 backend 和 func_name
2. GET /op/<backend>/func/<func_name> → 函数签名，获得形参列表
3. 从 CALL 指令的读参数中提取实参: [n,-2]=/models/A, [n,-3]=/models/B, ...
4. 建立形参→实参映射: {A: /models/A, B: /models/B, alpha: 1.0, ...}
5. 批量 MGET /op/<backend>/func/<func_name>/0, /op/.../1, ... (一次 Redis 往返)
6. 对每条编译层指令:
   a) 解析 dxlang 字符串: opcode + 读参数列表 + 写参数列表
   b) 形参替换: 将形参名替换为实参值
   c) 展开为执行层 key: [i,0]=opcode, [i,-1]=param1, [i,1]=out1, ...
   d) 批量 SET 到子栈路径 /vthread/<vtid>/[n,0]/
7. VM 设置 PC 指向子栈第一条指令: SET pc = "[n,0]/[0,0]"
```

**复制规则：**

- 编译层内部的 `./` 相对路径参数，保持 `./` 形式不变
- 编译层引用外部堆变量的形参，替换为调用者传入的实参
- 立即数形参替换为具体数值
- 命名槽位（如 `./mm`）的 key 在 `/vthread/<vtid>/` 下创建（平级），不受子栈嵌套影响

### 6.3 嵌套调用

func A CALL func B，func B CALL func C，形成多层嵌套：

```
/vthread/<vtid>/[0,0]/                  ← A 调用 B 的子栈
/vthread/<vtid>/[0,0]/[3,0]/            ← B 调用 C 的子栈 (假设 B 的第3条指令是 call)
```

最大嵌套深度由实现定义（建议 ≥ 32）。

### 6.4 RETURN 语义

当 func 执行完毕（到达指令序列末尾，或遇到 `return` 指令）：

1. 返回值写入调用者 CALL 指令指定的写参数槽位
2. 子栈的 KV 路径被 DELETE（递归删除所有子 key）
3. PC 恢复到调用者 CALL 指令的下一条指令（addr0 + 1）

## 7. 异步执行模型

元程的执行是全异步的。VM 不直接执行计算，而是将计算指令分发给 op-plat。

### 7.1 执行循环

```
VM 循环:
  1. GET /vthread/<vtid> → {pc: "[n,0]", status: "running", ...}
  2. 读取 /vthread/<vtid>/[n,0] → opcode
  3. 判断指令类型:
     a) 张量计算指令 (add, matmul, relu, ...):
        → 将指令 + 参数打包，PUSH 到目标 op-plat 的命令队列
        → SET /vthread/<vtid> = {pc: "[n,0]", status: "wait", ...}
        → BLPOP vthread 的完成通知队列
        → op-plat 计算完毕，LPUSH 完成事件
        → VM 醒来，SET /vthread/<vtid> = {pc: "[n+1,0]", status: "running", ...}
        
     b) 控制流指令 (call, if, for, return):
        → VM 直接处理 (无需 op-plat)
        → call: 读取 /op/<backend>/func/<name> (编译层), eager 翻译到子栈
        → 更新 /vthread/<vtid> 的 pc 字段
        
     c) 生命周期指令 (newtensor, deltensor):
        → PUSH 到 heap-plat 的命令队列
        → 轻量操作，通常同步完成
```

### 7.2 异步通知机制

```
VM 与 op-plat 之间的通信:

  VM                            op-plat 命令队列                   op-plat
  ──                            ──────────────                    ───────
  PUSH cmd:<op-plat-id>  ───→  [cmd1, cmd2, cmd3]  ───→  RPOP 消费
                                                                    │
                                                               GPU 计算
                                                                    │
  BLPOP done:<vtid>  ←───  [done_event]  ←──────────────  LPUSH 完成事件
  │
  醒来，PC++
```

### 7.3 批量发射

VM 可分析指令序列，将连续的多条无依赖指令批量 PUSH 到 op-plat 队列，减少往返。

```
无依赖序列: [add(a,b)->c, mul(d,e)->f, relu(g)->h]
  → 三条指令的输入互不依赖对方的输出
  → VM 一次 PUSH 三条，op-plat 可并行或流水线执行
  → 三条全部完成后，一次 LPUSH 通知
```

## 8. op-plat 抽象契约

op-plat 是执行张量计算指令的被动进程。

### 8.1 必须实现

| 能力 | 说明 |
|------|------|
| 指令消费 | 从指定的通知队列 RPOP 消费指令 |
| 张量计算 | 执行至少一种张量运算（如 elementwise、matmul、reduce） |
| 完成通知 | 计算完成后 LPUSH 完成事件到指定队列 |
| Tensor 访问 | 根据 tensor 元信息中的物理地址，访问外部存储中的实际数据 |

### 8.2 指令格式

```
op-plat 消费的指令包含:
  {
    "opcode": "add",              ← 操作码
    "vtid": "1",                  ← 所属 vthread
    "pc": "[3,0]",                ← 对应 vthread 中的指令坐标
    "inputs": [                   ← 输入 tensor 元信息
      {"key": "/vthread/1/a", "dtype": "f32", "shape": [1024], 
       "address": {"node": "n1", "device": "gpu0", "shm": "/deepx_t_xxx"}}
    ],
    "outputs": [                  ← 输出 tensor 元信息
      {"key": "/vthread/1/c", "dtype": "f32", "shape": [1024],
       "address": {"node": "n1", "device": "gpu0", "shm": "/deepx_t_yyy"}}
    ]
  }
```

### 8.3 完成通知格式

```
op-plat 计算完成后:
  LPUSH done:<vtid> {
    "pc": "[3,0]",
    "status": "ok",
    "outputs_updated": ["/vthread/1/c"]
  }
```

## 9. heap-plat 抽象契约

heap-plat 管理 tensor 的生命周期。

### 9.1 必须实现

| 能力 | 说明 |
|------|------|
| Tensor 创建 | 分配外部存储空间，返回物理地址 |
| Tensor 删除 | 释放外部存储空间 |
| Tensor 克隆 | 在指定设备上创建 tensor 副本 |
| 元信息查询 | 根据 key 返回 tensor 的 dtype、shape、物理地址 |

### 9.2 指令格式

```
heap-plat 消费的指令:
  newtensor:  {op: "newtensor", key: "/models/weights", dtype: "f32", shape: [1024,512], device: "gpu0"}
  deltensor:  {op: "deltensor", key: "/models/weights"}
  clone:      {op: "clone", src: "/models/weights", dst: "/models/weights_copy", device: "gpu1"}
```

## 10. 生命周期

### 10.1 Vthread 生命周期

```
CREATE:
  pysdk 或编译器在 /vthread/ 下创建新的 vtid 子树
  SET /vthread/<vtid> = {"pc": "[0,0]", "status": "init"}
  写入入口指令 (call main)

EXECUTE:
  VM 拾取 status="init" 的 vthread
  进入执行循环 (§7.1)
  SET /vthread/<vtid> = {..., "status": "running"}

WAIT:
  张量计算指令发射后
  SET /vthread/<vtid> = {..., "status": "wait"}
  VM 阻塞在完成通知队列上

ERROR:
  指令执行失败
  SET /vthread/<vtid> = {..., "status": "error", "error": "..."}

DONE:
  vthread 执行完毕
  SET /vthread/<vtid> = {..., "status": "done"}
  VM 清理 /vthread/<vtid>/ 子树 (GC)
```

### 10.2 堆变量生命周期

```
创建: newtensor → heap-plat 分配外部存储 → SET 元信息到堆路径
使用: vthread 通过堆路径引用 tensor
删除: deltensor → heap-plat 释放外部存储 → DELETE 堆路径

引用计数 (实现可选):
  多个 vthread 可能引用同一堆 tensor
  实现方可使用引用计数管理，refcount=0 时自动回收
```

---

# 第二部分：DeepX 实现（待起草）

> 本部分将定义元程模型在 DeepX 中的具体实现，包括：
> - Redis 作为 KV 空间的具体 key 路径约定
> - op-cuda / op-metal 的具体协议
> - heap-cuda / heap-metal 的具体协议
> - VM 进程的启动与调度实现
> - pysdk 与编译器的接口

---

## 附录 A：与 OS 进程的对照

| OS 进程概念 | 元程对应 |
|------------|---------|
| 虚拟地址空间 | KV 空间 |
| 进程 | 一个 KV 空间实例 |
| 线程 | Vthread |
| 代码段 (.text) | /src/func/ (源码) + /op/<backend>/func/ (编译) |
| 堆段 (.data/.bss) | 非保留路径 (堆变量) |
| 栈段 | /vthread/<vtid>/ 子树 |
| 程序计数器 (PC) | /vthread/<vtid> 的 value.pc 字段 |
| 栈帧 | CALL 产生的子维度 |
| 系统调用 | heap-plat / op-plat 命令 |
| 文件描述符 | 堆 tensor 引用 |
| 互斥锁 | LOCK/UNLOCK |

## 附录 B：与 x86 指令的对照

| x86 指令 | 元程等价 |
|----------|---------|
| `mov [addr], value` | `SET /heap/x = value` |
| `add eax, ebx` | `add(./a, ./b) -> ./c` |
| `push rax` | 写入命名槽位 |
| `call func` | `call` 指令 → 复制 func 到子栈 |
| `ret` | RETURN → 删除子栈 |
| `jmp label` | PC 跳转 |
| `cmp + jcc` | `if(cond) ->` |
| `int 0x80` | PUSH 到 op-plat 命令队列 |

## 附录 C：术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 元程 | Metaproc | 一个 KV 空间实例，分布式计算的边界 |
| 元线程 | Vthread | 元程内的执行流 |
| 函数 | Func | /src/func/ 下的可复用代码单元 (编译器优化后→/op/<backend>/func/) |
| 堆变量 | Heap Variable | 全局共享的长生命周期数据 |
| 栈帧 | Stack Frame | CALL 产生的子维度 |
| 二维寻址 | 2D Addressing | [addr0, addr1] 指令坐标系统 |
| 计算平面 | op-plat | 执行张量计算的后端进程 |
| 堆平面 | heap-plat | 管理 tensor 生命周期的后端进程 |
