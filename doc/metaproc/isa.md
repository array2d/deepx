# 元程指令集架构 (Metaproc ISA, MISA)

> 传统 CPU 指令集操作的是寄存器和内存地址。元程指令集操作的是 **KV 路径**和 **命名槽位**。
> 每条指令由 VM 调度器解释，分发到具备相应算子能力的 executor 执行。

## 1. 与传统 ISA 的根本差异

| 维度 | x86/ARM ISA | MISA |
|------|------------|------|
| 寻址方式 | 寄存器 + 立即数 + 内存地址 (int) | KV 路径 + 命名槽位 + 前缀解析 |
| 操作数类型 | int/float (标量) | tensor (多维数组) |
| 指令粒度 | 单周期操作 (add rax, rbx) | 算子调用 (matmul A, B → C) |
| 执行位置 | 固定 CPU 核 | 分布式 executor 池 |
| 地址空间 | 64 位虚拟地址 | 无限 KV 命名空间 |
| 调用开销 | ~10 cycles | 网络往返 + GPU kernel |
| 并行度 | ILP + SIMD | 算子内并行 + 元程间并行 |

## 2. 指令格式

```
┌──────────┬──────────────────────────────────────────────┐
│  opcode  │  operands (KV路径 / 命名槽位 / 立即数)        │
├──────────┼──────────────────────────────────────────────┤
│  flags   │  {async, device, priority, timeout}           │
└──────────┴──────────────────────────────────────────────┘
```

**操作数类型：**

| 前缀 | 含义 | 示例 | 解析为 |
|------|------|------|--------|
| `$name` | 命名槽位（形参） | `$input` | 调用时绑定的 tensor 路径 |
| `heap:name` | 堆 tensor | `heap:weights` | `/vproc/self/heap/weights` |
| `stack:name` | 栈 tensor | `stack:hidden` | `/vproc/self/stack/hidden` |
| `/path` | 绝对 KV 路径 | `/tensor/model/w` | 全局路径 |
| `@vpid:name` | 其他元程的堆 | `@42:weights` | `/vproc/42/heap/weights` |
| `#literal` | 立即数 | `#3.14` | 标量值 |

## 3. 指令集全表

### 3.1 张量生命周期类 (Tensor Lifecycle)

```
TENSOR_NEW   heap:name, dtype, shape[, device]
  在堆区分配 tensor。通知 heapmem 分配 shm + GPU buffer。
  等价于: OS 的 malloc() + mmap()

TENSOR_DEL   heap:name
  堆 tensor 引用计数 -1。若为 0，回收 shm。
  等价于: OS 的 free() + munmap()

TENSOR_TEMP  stack:name, dtype, shape[, device]
  在栈区分配临时 tensor。带 TTL，代码块结束自动回收。
  等价于: OS 的 alloca() / 栈分配

TENSOR_PROMOTE stack:name → heap:name
  将栈 tensor 升级为堆 tensor（如 checkpoint 保存）。
  等价于: 栈数据 malloc + memcpy 到堆

TENSOR_CLONE  heap:src → heap:dst
  深拷贝堆 tensor。新 shm，新 refcount=1。
  等价于: OS 的 fork() 中的 COW 页复制
```

### 3.2 数据移动类 (Data Movement)

```
TENSOR_LOAD  heap:name → $slot[, range]
  将 tensor 数据加载到 executor 本地 GPU buffer。
  若已在目标设备上，跳过。
  等价于: OS 的 LOAD 指令 (内存→寄存器)

TENSOR_STORE $slot → heap:name[, range]
  将计算结果写回堆 tensor。更新 version。
  等价于: OS 的 STORE 指令 (寄存器→内存)

TENSOR_MOVE  heap:name, device
  将 tensor 迁移到指定设备 (GPU0→GPU1, GPU→CPU)。
  等价于: NUMA 的页迁移

TENSOR_CAST  heap:name, dtype
  类型转换 (f32→f16)。可能原地或新建。
```

### 3.3 张量计算类 (Tensor Compute)

```
TENSOR_OP   opname, inputs..., outputs..., [flags]
  核心计算指令。将 inputs tensor 送入算子 opname，结果写入 outputs。
  
  opname 来自 /op/ 全局注册表:
    elementwise: add, sub, mul, div
    activation:  relu, sigmoid, tanh, gelu
    matmul:      matmul, batch_matmul
    reduce:      sum, mean, max, min
    changeshape: reshape, transpose, concat, slice
    init:        zeros, ones, arange, random
    io:          save, load

  等价于: OS 的 ALU 指令，但操作数是多维 tensor
```

### 3.4 控制流类 (Control Flow)

```
JMP        code_block
  跳转到指定代码块执行。
  等价于: OS 的 JMP 指令

COND       condition, then_block, else_block
  条件分支。condition 基于 metrics 中的标量值。
  等价于: OS 的 JZ / JNZ / CMP + Jcc

CALL       @vpid:code_block, bindings
  调用另一元程的代码块。等待返回。
  bindings: {$param: path, ...}
  等价于: OS 的 CALL 指令 + ABI 传参

RETURN     $output...
  从当前代码块返回，将 $output 绑定到调用者的命名槽位。
  等价于: OS 的 RET 指令

YIELD
  让出执行权。将元程状态置为 ready，允许其他元程被调度。
  等价于: OS 的 sched_yield()

LOOP       count, code_block
  循环执行代码块 count 次。
  等价于: OS 的 LOOP / REP 前缀
```

### 3.5 同步类 (Synchronization)

```
LOCK       /tensor/name/lock[, timeout]
  获取全局 tensor 排他锁。超时则进入 error 状态。
  等价于: OS 的 mutex_lock()

UNLOCK     /tensor/name/lock
  释放全局 tensor 锁。
  等价于: OS 的 mutex_unlock()

BARRIER    barrier_name, count
  等待 count 个元程到达屏障点。
  等价于: MPI_Barrier / pthread_barrier

WATCH      /path, on_change → code_block
  注册 KV 路径的监听回调。路径值变化时触发指定代码块。
  等价于: OS 的信号处理 / inotify

UNWATCH    /path
  取消监听。
```

### 3.6 系统类 (System)

```
HALT
  停止元程执行。状态 → stopped。
  等价于: OS 的 exit() / HLT

CHECKPOINT heap:name..., path
  将堆 tensor 持久化到文件。
  等价于: OS 的 fsync() + write()

GC
  触发垃圾回收。遍历 heap/，refcount=0 的回收 shm。
  等价于: OS 的内存回收 / OOM killer 的优雅版

NOOP
  空操作。用于占位或对齐。
  等价于: OS 的 NOP
```

## 4. 指令编码（dxlang 格式）

```
{
  "lang": "dxlang",
  "version": 1,
  "meta": {"author": "train_mnist"},
  "seq": [
    {"op": "TENSOR_NEW",  "name": "heap:weights", "dtype": "f32", "shape": [784,256]},
    {"op": "TENSOR_NEW",  "name": "heap:bias",    "dtype": "f32", "shape": [256]},
    {"op": "TENSOR_TEMP", "name": "stack:hidden",  "dtype": "f32", "shape": [1024,256]},
    {"op": "TENSOR_OP",   "opname":"matmul", "A":"$input","B":"heap:weights","C":"stack:hidden"},
    {"op": "TENSOR_OP",   "opname":"add",    "A":"stack:hidden","B":"heap:bias","C":"stack:hidden2"},
    {"op": "TENSOR_OP",   "opname":"relu",   "A":"stack:hidden2","C":"$output"},
    {"op": "COND",        "condition": "loss < 0.01",
     "then": "checkpoint", "else": "backward"}
  ]
}
```

## 5. 指令执行模型

```
传统 CPU 执行:
  PC → 取指 → 译码 → 执行 → 写回 → PC++

元程 VM 执行:
  current_op_idx → 取指 (GET code/<block>.seq[idx])
  → 译码 (识别 opcode + 解析 KV 路径)
  → 路由 (GET /op/<opname>/impl → 选择 executor)
  → 分发 (UDP 发送任务包到 executor)
  → 等待 (异步回调或同步阻塞)
  → 写回 (SET status.current_op_idx += 1)
  → 检查 (是否 seq 末尾? → 切换代码块)
```

## 6. 指令级并行

传统 CPU 用流水线和超标量实现 ILP。
元程的并行策略：

```
流水线并行:
  取指 (VM) │ 译码 (VM) │ 路由 (VM) │ 执行 (executor) │ 写回 (VM)
  ──────────────────────────────────────────────────────────→
  指令 N+1 的取指/译码/路由 可与 指令 N 的执行 重叠

数据并行:
  单条 TENSOR_OP 指令内部，GPU 用数千线程并行计算

元程间并行:
  多个元程的指令流被 VM 交替调度，类似 OS 的抢占式多任务
```

## 7. 与传统 ISA 的精妙对应

```
x86 MOV rax, [rbx]      →  TENSOR_LOAD  heap:weights → $slot
x86 ADD rax, rbx        →  TENSOR_OP add, A, B → C
x86 PUSH rax            →  TENSOR_TEMP stack:x
x86 CALL func           →  CALL @dataloader:next_batch
x86 CMP rax, 0; JZ lbl  →  COND loss<0.01, then, else
x86 INT 0x80            →  WATCH /tensor/X (事件中断)
x86 HLT                 →  HALT
```

每条 MISA 指令不是操作一个标量，而是操作一个 **全局可寻址的 tensor**。
KV 路径代替了 64 位整数地址，让寻址空间从单机 2^64 扩展到整个 deepx 集群的 KVSpace。
