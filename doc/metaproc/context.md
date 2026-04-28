# 元程上下文 (Metaproc Context Block, MCB)

> 传统 OS 用 PCB (Process Control Block) 保存进程的所有状态，
> 以便内核在进程间切换时恢复执行现场。
> 元程的 MCB 存于 KVSpace 的 `context/` 子树中，
> 由 VM 调度器读取和更新，支持跨机器、跨代码块的执行现场保存与恢复。

## 1. PCB vs MCB

| PCB 字段 (Linux task_struct) | MCB 对应 | 存储位置 |
|------------------------------|---------|---------|
| 通用寄存器 (rax, rbx...) | 命名槽位绑定表 | `context/registers` |
| 程序计数器 (RIP) | 当前代码块 + 指令索引 | `status.current_code`, `status.current_op_idx` |
| 栈指针 (RSP, RBP) | 栈帧链表 | `context/stack_frames` |
| 页表基址 (CR3) | 路径表根 (vpid) | 隐式: `/vproc/<vpid>/` |
| 打开文件表 | tensor 引用表 | `context/tensor_refs` |
| 信号处理表 | WATCH 回调表 | `context/watch_list` |
| 进程状态 (TASK_RUNNING...) | 元程状态 | `status.state` |
| CPU 亲和性 | 设备亲和性 | `config/default_device` |
| 命名空间 (nsproxy) | KV 路径根 | `/vproc/<vpid>/` 子树 |
| 内存描述符 (mm_struct) | heap/ + stack/ 子树 | 由 heapmem 管理 |
| 锁持有列表 | lock set | `context/locks_held` |

## 2. MCB 的 KVSpace 布局

```
/vproc/<vpid>/context/
│
├── registers                    命名槽位绑定表 (寄存器文件)
│   { "$input":   "stack:batch",
│     "$output":  "stack:logits",
│     "$temp0":   "stack:intermediate",
│     "$loss":    "#0.0" }
│
├── stack_frames                 栈帧链表 (调用栈)
│   [ { "code": "train_epoch",
│       "bindings": { "$epoch": "#10", "$model": "heap:weights" },
│       "return_to": null,
│       "local_slots": ["$batch", "$loss"] },
│     { "code": "train_step",
│       "bindings": { "$batch": "stack:batch", "$model": "heap:weights" },
│       "return_to": "train_epoch",
│       "local_slots": ["$hidden", "$grad"] } ]
│
├── tensor_refs                  tensor 引用表 (文件描述符表等价)
│   { "/tensor/model/weights":   { "ref_mode": "read",  "version_known": 5 },
│     "/tensor/rl/state":        { "ref_mode": "write", "version_known": 3 },
│     "/tensor/dataset/cifar10": { "ref_mode": "read",  "version_known": 1 } }
│
├── watch_list                   WATCH 回调注册表
│   [ { "path":      "/tensor/rl/state",
│       "event":     "version_change",
│       "callback":  "on_state_update",
│       "arg":       { "$new_state": "stack:rl_state" } },
│     { "path":      "/vproc/dataloader/status",
│       "event":     "state=ready",
│       "callback":  "on_batch_ready",
│       "arg":       {} } ]
│
├── locks_held                   当前持有的锁
│   [ { "path": "/tensor/model/weights/lock",
│       "acquired_at": 1714000000,
│       "timeout_at": 1714000030 } ]
│
├── call_stack                   嵌套 CALL 的返回地址栈
│   [ { "caller_vpid": 1,
│       "caller_code": "train_epoch",
│       "return_slot": "$metrics",
│       "bind_back": { "$metrics": "stack:epoch_result" } } ]
│
├── pc_history                   程序计数器历史 (用于调试和回滚)
│   [ {"code":"forward",  "op_idx":0, "op":"matmul",      "at":1714000001},
│     {"code":"forward",  "op_idx":1, "op":"add",         "at":1714000010},
│     {"code":"forward",  "op_idx":2, "op":"relu",        "at":1714000012} ]
│
├── version_vector               向量时钟 (用于缓存一致性)
│   { "/vproc/self/heap/weights": 5,
│     "/tensor/rl/state":         3,
│     "/vproc/dataloader/heap/batch": 128 }
│
└── exception_frame               异常处理帧 (当前错误处理上下文)
    { "on_error": "error_handler",
      "retry_count": 3,
      "max_retries": 5,
      "last_error": "GPU OOM at op_idx 3" }
```

## 3. 命名槽位 — 寄存器文件

在传统 CPU 中，寄存器是固定数量 (如 16 个通用寄存器) 的快速存储。
在元程中，**命名槽位**是逻辑概念，数量不固定，但其"活跃集"有上限。

```
传统寄存器:
  rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15
  
  指令: mov rax, [0x1000]
        add rax, rbx
        push rax

元程命名槽位:
  $input, $output, $temp0, $batch, $loss, $hidden...
  
  指令: {"op":"TENSOR_LOAD", "target":"$slot", "source":"heap:weights"}
        {"op":"TENSOR_OP",   "opname":"matmul", "A":"$slot", ...}
        {"op":"TENSOR_TEMP", "name":"stack:hidden", ...}
```

### 3.1 槽位绑定与生命周期

```
┌─ 代码块入口 ──────────────────────────────────────────────────┐
│                                                                │
│  调用者: CALL @42:forward, bind={ "$input": "stack:batch" }   │
│                                                                │
│  执行时:                                                       │
│    1. VM 将 "$input" 绑定到 "/vproc/42/stack/batch"           │
│    2. 写入 context/registers["$input"] = "stack:batch"        │
│    3. 执行代码块                                              │
│    4. RETURN 时，解绑所有命名槽位                              │
│                                                                │
│  绑定规则:                                                     │
│    - $input  → 调用者指定的路径 (可以是 heap:/stack:/绝对路径) │
│    - $output → 代码块产出，RETURN 传回调用者                   │
│    - $temp*  → 代码块内部使用，代码块结束时释放                │
│    - $#literal → 常量，不绑定路径                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 调用者保存 vs 被调用者保存

类似 x86 ABI 的 caller-saved / callee-saved 寄存器约定：

```
调用者保存 (caller-saved):
  $temp0, $temp1, $temp2 ...
  被调用代码块可以自由修改
  调用者如需保留，应在 CALL 前将其值写入 heap/stack

被调用者保存 (callee-saved):
  $model, $config, $device ...
  被调用代码块必须在使用后恢复原值
  类似 x86 的 rbx, rbp, r12-r15

参数槽位 (parameter slots):
  $input, $output
  特殊：$input 由调用者设置，被调用者只读
        $output 由被调用者写入，调用者读取
```

## 4. 程序计数器 (PC)

传统 CPU 的 PC 是一个整数，指向当前指令在代码段中的偏移。

元程的 PC 是一个**二维坐标**：

```
status = {
    "state":          "running",       ← 整体状态
    "current_code":   "forward",       ← 当前代码块 (段选择子)
    "current_op_idx": 3,               ← 当前指令在 seq 中的索引 (偏移)
    "next_code":      "backward",      ← 当前代码块完成后跳转到哪
    "caller_code":    null,            ← 如果被 CALL，调用者是谁
    "caller_op_idx":  null             ← 调用者的指令索引 (返回地址)
}
```

### 4.1 PC 推进规则

```
顺序执行:
  TENSOR_OP 完成 → current_op_idx += 1

条件分支:
  COND condition met    → current_code = then_block, op_idx = 0
  COND condition unmet   → current_code = else_block, op_idx = 0

调用:
  CALL @target:code       → push 当前 PC 到 call_stack
                          → current_code = code, op_idx = 0

返回:
  RETURN                  → pop call_stack, 恢复调用者的 PC

异常:
  算子执行失败            → current_code = error_handler
                          → exception_frame 记录现场

代码块结束:
  seq 全部执行完          → current_code = next_code (如果有)
                          → 否则 state = "ready"
```

## 5. 栈帧 (Stack Frames)

传统进程的栈帧：
```
高地址 ┌────────────┐
       │  参数 N     │
       │  ...        │
       │  参数 1     │
       ├────────────┤
       │  返回地址   │  ← CALL 压入
       ├────────────┤
       │  旧 RBP    │  ← push rbp
RBP →  ├────────────┤
       │  局部变量   │
RSP →  └────────────┘ 低地址
```

元程的栈帧在 KVSpace 中，是一棵逻辑树：

```
调用链:
  train_epoch
    └─ CALL dataloader:next_batch   ← 栈帧#1
         └─ CALL preprocessor:normalize ← 栈帧#2

context/stack_frames = [
  { "depth": 0,
    "code": "train_epoch",
    "bindings": { "$epoch": "#10" },
    "locals": ["$batch", "$loss"],
    "return_to": null },

  { "depth": 1,
    "code": "next_batch",
    "bindings": { "$batch_size": "#64" },
    "locals": ["$indices", "$data"],
    "return_to": "train_epoch",
    "return_idx": 3 },

  { "depth": 2,
    "code": "normalize",
    "bindings": { "$data": "stack:raw_data" },
    "locals": ["$mean", "$std"],
    "return_to": "next_batch",
    "return_idx": 2 }
]
```

### 5.1 栈上的临时 tensor

```
栈帧 #2 (normalize) 中的临时变量 $mean:
  → TENSOR_TEMP stack:mean, f32, [256]
  → /vproc/42/stack/mean = {dtype:"f32", shape:[256], TTL:300}

normalize 代码块 RETURN 后:
  → VM 清理该帧所有 stack tensor
  → 释放 shm
  → 清理 /vproc/42/stack/mean
```

## 6. WATCH 回调表

类似 OS 的信号处理表，但更结构化：

```
OS 信号:
  signal(SIGINT, handler)
  → 内核在进程收到 SIGINT 时调用 handler

元程 WATCH:
  WATCH /tensor/rl_state, on_change → code="on_state_update"
  → VM 在 /tensor/rl_state 变化时调度该代码块
```

```
WATCH 事件类型:
  value_change      值被 SET (version 变化)
  child_added       子节点新增 (如新的 heap tensor)
  child_removed     子节点删除
  lock_acquired     锁被其他元程获取
  lock_released     锁释放
  state_transition  元程状态变化 (init→ready→running)
  timeout           超时 (带 TTL 的路径过期)
```

```
WATCH 触发时的调度:

1. VM 收到 Redis PUBLISH: /tensor/rl_state updated
2. 遍历所有元程的 context/watch_list
3. 元程 42 的 watch_list 中有匹配项:
     {path:"/tensor/rl_state", event:"value_change", callback:"on_state_update"}
4. VM 检查元程 42 当前状态:
   - 如果 state=wait → 将 state 置为 ready, next_code = "on_state_update"
   - 如果 state=ready → 排队 "on_state_update" 到待执行队列
   - 如果 state=running → 标记 pending_callback，当前代码块完成后执行
5. 下次 VM 调度元程 42 时，执行 on_state_update
```

## 7. 上下文切换 (Context Switch)

传统 OS 的上下文切换：
```
  时钟中断 → 保存寄存器 → 切换页表 → 加载新进程寄存器 → iret
  开销: ~1-10 μs
```

元程的上下文切换发生在 VM 调度器**将执行权从一个代码块转移到另一个**时：

### 7.1 代码块间切换 (轻量)

```
同一元程内，当前代码块完成 → 下一个代码块:

1. 当前代码块 seq 全部执行完
2. 检查 next_code:
   - 如果 next_code = "backward" → 加载 code/backward
3. 保存当前帧信息:
   - pc_history 追加当前代码块的执行记录
4. 切换到新代码块:
   SET status.current_code = "backward"
   SET status.current_op_idx = 0
5. 开始执行新代码块的第一条指令

开销: ~100 μs (几次 Redis 操作)
```

### 7.2 元程间切换 (重量)

```
元程 A 暂停 / 元程 B 开始执行:

1. VM 决定切换 (如 A 进入 wait, B 处于 ready)
2. 保存 A 的 MCB:
   - context/registers 快照
   - status.current_code + current_op_idx 已是最新
3. 加载 B 的 MCB:
   - GET /vproc/B/context/registers
   - GET /vproc/B/status
   - GET /vproc/B/code/<current_code>
4. 路由 B 的下一条指令到 executor

开销: ~1-5 ms (取决于 cache 命中)
```

### 7.3 抢占 vs 协作

```
OS: 抢占式调度。时钟中断强制切换。
元程: 协作式调度为主，抢占为辅。

协作点:
  - TENSOR_OP 执行完成 (自然地让出)
  - YIELD 指令 (显式让出)
  - COND 分支点
  - LOCK 等待超时

抢占:
  - VM 可以 SET status.state = "paused"
  - executor 收到暂停信号后完成当前 kernel 并报告
  - 紧急情况 (如更高优先级元程就绪)
```

## 8. 异常与错误处理

### 8.1 异常帧

```
exception_frame: {
    "on_error":    "error_handler",   ← 错误处理代码块
    "retry_count": 3,
    "max_retries": 5,
    "last_error": {
        "code":     "OOM",
        "message":  "GPU buffer allocation failed: requested 2GB, available 1.5GB",
        "op":       "TENSOR_NEW",
        "op_idx":   3,
        "at":       1714000100
    },
    "error_history": [...]            ← 最近 N 条错误
}
```

### 8.2 错误恢复路径

```
TENSOR_OP 执行失败:
  1. executor 返回 error
  2. VM 记录错误到 exception_frame
  3. 检查 retry_count < max_retries?
     YES → retry_count++, 重新分发指令 (可能换一个 executor)
     NO  → 检查 on_error 代码块是否存在?
            YES → 跳转到 error_handler (类似 OS 的异常处理)
            NO  → state = "error", 等待人工介入
```

## 9. 元程状态与上下文的关联

```
status.state          context/ 中的关键变化
─────────────         ──────────────────────
init                  清空 registers, 加载 code/init
ready                 pc_history 清空, call_stack 为空
running               PC 持续推进, stack_frames 可能增长
wait                  current_code = null, WATCH 等待触发
error                 exception_frame 记录错误, PC 停在出错位置
stopped               context/ 子树标记为待 GC

状态转换时的 MCB 操作:

ready → running:
  SET status.state = "running"
  SET status.current_code = next_code (或第一个代码块)
  SET status.current_op_idx = 0

running → wait:
  SET status.state = "wait"
  保持 status.current_code + current_op_idx (恢复点)
  WATCH 注册到 context/watch_list 中指定的路径

wait → ready:
  SET status.state = "ready"
  SET status.next_code = "on_state_update" (从 WATCH 回调)
  context/watch_list 中的匹配项标记为 triggered

running → error:
  SET status.state = "error"
  写入 status.error = last_error
  exception_frame 记录完整现场
```

## 10. 与传统 PCB 的精妙对应总结

```
Linux task_struct             元程 context/ 子树
────────────────              ──────────────────
state (TASK_RUNNING...)       status.state
pid                           vpid (隐含)
stack (vm_area)               stack_frames
mm_struct → pgd               vpid → /vproc/<vpid>/ (路径表根)
thread_struct (寄存器快照)     context/registers
thread.ip (RIP)               status.current_op_idx
thread.sp (RSP)               stack_frames[-1] (栈顶帧)
files_struct                  context/tensor_refs
sighand_struct                context/watch_list
held_locks                    context/locks_held
nsproxy                       /vproc/<vpid>/ 子树 (命名空间隔离)
seccomp                       method/ 算子能力 (允许执行的指令子集)
ptrace                        WATCH /vproc/<vpid>/status (外部监控)
```

一个 OS 进程的 PCB 约 1-10 KB，存于内核内存。
一个元程的 MCB 约 1-10 KB JSON，存于 Redis KVSpace。

二者在功能上完整对应：都是保存程序执行现场的数据结构，都是调度器在切换执行单元时必须读写的关键状态。
