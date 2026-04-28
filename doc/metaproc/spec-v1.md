# 元程规范 (Metaproc Specification) v1

> 元程是一种基于 KV 寻址空间的分布式计算模型。
> 本规范定义元程的抽象语义，不依赖任何具体实现（Redis、CUDA、Metal 等）。
> 实现方只需提供一个满足 §2 要求的 KV 空间，即可运行元程程序。

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

定义在 `/funcs/` 下的可复用代码单元。由指令序列组成。
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
| `/vthread/` | 所有 vthread 的执行状态 |
| `/funcs/` | 全局函数定义 |
| `/sys/` | 系统信息（op-plat、heap-plat 注册等） |

### 3.2 堆的隐式命名空间

除保留路径外，KV 空间中的所有其他路径均为堆路径。
堆路径的具体命名空间划分由上层（如 pysdk）自行管理。

```
例:
  /models/bert/weights    ← 堆
  /data/cifar10/train     ← 堆
  /checkpoints/step1000   ← 堆
  /vthread/1/...          ← 栈 (保留)
  /funcs/forward          ← 函数 (保留)
```

### 3.3 相对路径

vthread 栈内使用 `./` 前缀表示相对路径，解析基点为当前栈帧。

```
./mm       → 当前栈帧下的局部变量 mm
./bias     → 当前栈帧下的局部变量 bias
../a       → 上一级栈帧的局部变量 a (不推荐，但技术上可行)
```

## 4. 函数定义

### 4.1 路径结构

```
/funcs/<func_name>          → 函数签名 (类型信息)
/funcs/<func_name>/0        → 第 0 条指令
/funcs/<func_name>/1        → 第 1 条指令
/funcs/<func_name>/2        → 第 2 条指令
...
```

### 4.2 函数签名

`/funcs/<func_name>` 的值定义函数的类型签名：

```
(func_name((ro_p1:type1, ro_p2:type2, ...) -> (w_p1:type3, w_p2:type4, ...))
```

- 左侧 `()` 内为只读参数
- 右侧 `()` 内为写入参数（返回值）

### 4.3 指令

`/funcs/<func_name>/N` 的值是一条指令。

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
  /funcs/<name>/N/true/0      ← true 分支第一条指令
  /funcs/<name>/N/false/0     ← false 分支第一条指令

for(iterator) ->               ← 循环入口
  /funcs/<name>/N/body/0      ← 循环体第一条指令
```

### 4.4 示例

```
/funcs/gemm = (gemm(A:tensor<f32>, B:tensor<f32>, alpha:f32, beta:f32, C:tensor<f32>) -> (Y:tensor<f32>))

/funcs/gemm/0 = matmul(A, B) -> ./Y
/funcs/gemm/1 = mul(./Y, alpha) -> ./Y
/funcs/gemm/2 = mul(C, beta) -> ./C
/funcs/gemm/3 = add(./Y, ./C) -> ./Y
```

## 5. Vthread 执行模型

### 5.1 Vthread 路径结构

```
/vthread/<vtid>/[0,0]        ← 栈帧 #0，指令 #0 的操作码
/vthread/<vtid>/[0,-1]       ← 栈帧 #0，指令 #0 的读参数 #1
/vthread/<vtid>/[0,-2]       ← 栈帧 #0，指令 #0 的读参数 #2
/vthread/<vtid>/[0, 1]       ← 栈帧 #0，指令 #0 的写参数 #1
/vthread/<vtid>/[1,0]        ← 栈帧 #0，指令 #1 的操作码
...
/vthread/<vtid>/a            ← 栈帧 #0 的命名槽位 a 的值
/vthread/<vtid>/b            ← 栈帧 #0 的命名槽位 b 的值
/vthread/<vtid>/pc           ← 当前程序计数器 (如 [2,0])
/vthread/<vtid>/status       ← vthread 状态
```

### 5.2 二维寻址

vthread 的指令序列使用二维坐标 `[addr0, addr1]` 寻址：

| addr1 | 含义 | 示例 |
|-------|------|------|
| `0` | 操作码 | `call`, `add`, `matmul`, `newtensor` |
| `-1, -2, ...` | 读取参数 (左值) | func 名、输入 tensor、立即数 |
| `1, 2, ...` | 写入参数 (右值) | 输出 tensor、返回值 |

addr0 是序列维，表示指令在栈帧内的顺序位置。

### 5.3 命名槽位

vthread 栈帧下可以有命名槽位（如 `/vthread/<vtid>/a`），用于存放局部变量的值。

基础类型（int, float, bool, 短 string）的值直接存储在槽位中。
Tensor 类型的值存储 tensor 元信息（dtype, shape, 物理地址）。

### 5.4 程序计数器 (PC)

`/vthread/<vtid>/pc` 的值指向当前待执行指令的坐标，如 `[3,0]`。

### 5.5 Vthread 状态

`/vthread/<vtid>/status` 的值：

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

### 6.2 子栈创建

VM 将 `/funcs/<func_name>/` 下的指令序列**复制**到 vthread 的子栈中，形成新的维度：

```
复制前:                           复制后:
/vthread/<vtid>/[n,0]             /vthread/<vtid>/[n,0]  = call
                                  /vthread/<vtid>/[n,0]/[0,0]  = <func 的第0条 opcode>
                                  /vthread/<vtid>/[n,0]/[0,-1] = <参数>
                                  /vthread/<vtid>/[n,0]/[0, 1] = <返回槽位>
                                  /vthread/<vtid>/[n,0]/[1,0]  = <func 的第1条 opcode>
                                  ...
```

复制规则：
- func 内部的 `./` 相对路径参数，保持为 `./` 形式不变（解析基点自动变为子栈）
- func 引用外部堆变量的绝对路径参数，保持原样
- 调用者传入的实参（如 `A`），替换 func 中对应的形参

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
  1. 读取 /vthread/<vtid>/pc → 当前坐标 [n,0]
  2. 读取 /vthread/<vtid>/[n,0] → opcode
  3. 判断指令类型:
     a) 张量计算指令 (add, matmul, relu, ...):
        → 将指令 + 参数打包，PUSH 到目标 op-plat 的命令队列
        → SET status = "wait"
        → BLPOP vthread 的完成通知队列
        → op-plat 计算完毕，LPUSH 完成事件
        → VM 醒来，SET pc = [n+1, 0]
        
     b) 控制流指令 (call, if, for, return):
        → VM 直接处理 (无需 op-plat)
        → 更新 PC 和栈结构
        
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
  SET status = "init"
  SET pc = [0,0]
  写入入口指令 (call main)

EXECUTE:
  VM 拾取 status="init" 的 vthread
  进入执行循环 (§7.1)
  status → "running"

WAIT:
  张量计算指令发射后，status = "wait"
  VM 阻塞在完成通知队列上

ERROR:
  指令执行失败 → status = "error"
  错误信息写入 /vthread/<vtid>/error

DONE:
  vthread 执行完毕 → status = "done"
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
> - heapmem-cuda / heapmem-metal 的具体协议
> - VM 进程的启动与调度实现
> - pysdk 与编译器的接口

---

## 附录 A：与 OS 进程的对照

| OS 进程概念 | 元程对应 |
|------------|---------|
| 虚拟地址空间 | KV 空间 |
| 进程 | 一个 KV 空间实例 |
| 线程 | Vthread |
| 代码段 (.text) | /funcs/ |
| 堆段 (.data/.bss) | 非保留路径 (堆变量) |
| 栈段 | /vthread/<vtid>/ 子树 |
| 程序计数器 (PC) | /vthread/<vtid>/pc |
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
| 函数 | Func | /funcs/ 下的可复用代码单元 |
| 堆变量 | Heap Variable | 全局共享的长生命周期数据 |
| 栈帧 | Stack Frame | CALL 产生的子维度 |
| 二维寻址 | 2D Addressing | [addr0, addr1] 指令坐标系统 |
| 计算平面 | op-plat | 执行张量计算的后端进程 |
| 堆平面 | heap-plat | 管理 tensor 生命周期的后端进程 |
