# 调用约定与跨元程通信 ABI

> 传统 CPU 有 ABI (Application Binary Interface)，定义函数调用时参数如何传递、
> 寄存器如何保存、返回值如何回传。元程需要一套对应的 **MABI (Metaproc ABI)**，
> 定义跨代码块、跨元程的调用约定，以及通过 /tensor/ 共享堆实现的进程间通信。

## 1. 从 x86 ABI 到 MABI

| x86-64 System V ABI | MABI 对应 |
|---------------------|----------|
| 参数传递: rdi, rsi, rdx, rcx, r8, r9 | 参数传递: `bind={ "$p1": path, "$p2": path, ... }` |
| 返回值: rax, rdx | 返回值: `return_slot: "$result"` |
| caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11 | caller-saved: `$temp0-9` |
| callee-saved: rbx, rbp, r12-r15 | callee-saved: `$model, $state, $config` |
| 栈帧: push rbp; mov rbp, rsp | 栈帧: context/stack_frames 追加条目 |
| CALL: push return_addr; jmp func | CALL: push call_stack; switch code_block |
| RET: pop rip | RETURN: pop call_stack; restore PC |
| 栈清理: caller (cdecl) 或 callee (stdcall) | 栈清理: VM 自动清理 (类似 stdcall，被调用者释放 stack tensor) |

## 2. 调用层次

元程的调用分为三个层次：

```
Level 1: 本地调用 (同元程内的代码块跳转)
  JMP backward
  COND loss<0.01 → checkpoint
  开销: ~100 μs (仅 VM 状态更新)

Level 2: 直接调用 (同机器内，夸元程)
  CALL @dataloader:next_batch, bind={...}
  等被调用元程执行完 code/next_batch，获取返回值
  开销: ~1-10 ms (Redis 同步 + tensor shm 共享)

Level 3: 远程调用 (跨机器)
  CALL @trainer_n2:forward, bind={...}
  VM 调度器协调跨机器 executors
  开销: ~10-100 ms (网络往返 + 可能的 tensor 迁移)
```

## 3. CALL 指令语义

### 3.1 指令格式

```json
{
  "op": "CALL",
  "target": {
    "vproc": "@dataloader",        // 被调用元程的 vpid 或符号名
    "code": "next_batch"           // 要执行的代码块
  },
  "bind": {
    "$batch_size": "#64",          // $参数名 → 值/路径
    "$dataset": "heap:cifar10"
  },
  "return": {
    "slot": "$batch",              // 返回值绑定到调用者的哪个命名槽位
    "timeout_ms": 30000            // 超时则进入 error
  },
  "flags": {
    "async": false,                // true=异步调用 (fire-and-forget)
    "priority": 1
  }
}
```

### 3.2 同步调用流程

```
调用者元程 A (vpid=1)                          被调用者元程 B (vpid=42)
═══════════════════════                        ═══════════════════════

1. 执行到 CALL @42:forward
   bind={ "$input": "stack:batch" }

2. VM 保存 A 的现场:
   push call_stack:
     { caller_vpid:1,
       caller_code:"train_epoch",
       caller_op_idx:5,
       return_slot:"$logits" }
                                             3. VM 检查 B 的状态:
   SET /vproc/1/status.state = "wait"          - 如果 B 处于 ready:
   SET /vproc/1/status.current_code = null       SET /vproc/42/status.state = "running"
                                                  SET /vproc/42/status.current_code = "forward"
                                                  SET /vproc/42/status.caller_code = "train_epoch"
                                                  SET /vproc/42/status.caller_op_idx = 5
                                             4. VM 绑定参数:
                                                SET /vproc/42/context/registers["$input"] = "stack:batch"
                                             5. B 开始执行 code/forward:
                                                TENSOR_OP matmul $input, heap:weights → stack:hidden
                                                TENSOR_OP add    stack:hidden, heap:bias → stack:hidden2
                                                TENSOR_OP relu   stack:hidden2 → $output

                                             6. B 执行到 RETURN $output

7. VM 收到 RETURN:
   - 将 $output 的路径复制到 A 的命名槽位 $logits
   - 即 SET /vproc/1/heap/logits 指向 B 产出的 tensor
     (如果不是共享的，则深拷贝)

8. VM 恢复 A 的现场:
   pop call_stack
   SET /vproc/1/status.state = "running"
   SET /vproc/1/status.current_code = "train_epoch"
   SET /vproc/1/status.current_op_idx = 6  (CALL 的下一条)

9. A 继续执行:
   TENSOR_OP softmax $logits → $probs
```

### 3.3 异步调用 (fire-and-forget)

```
CALL @logger:log_metrics, bind={ "$metrics": "stack:epoch_stats" },
     flags={ "async": true }

异步调用流程:
  1. VM 不保存 A 的现场到 call_stack
  2. A 不进入 wait 状态，继续执行下一条指令
  3. VM 将任务放入 B 的待执行队列
  4. B 在空闲时执行 log_metrics
  5. 没有返回值传递

适用场景:
  - 日志记录
  - 指标上报
  - 非关键的 side-effect 操作
  - 类似 x86 的 "发射后不管" 的 I/O 写操作
```

## 4. 参数传递约定

### 4.1 按引用传递 (tensor 路径)

```
默认行为：传递的是 KV 路径引用，不是数据副本。

CALL @42:forward, bind={ "$input": "stack:batch" }
→ B 看到 $input 绑定到 /vproc/1/stack/batch
→ B 的 executor 通过 shm_open 直接读取该 tensor
→ 不产生数据复制 (类似 C++ 的 const T&)
```

### 4.2 按值传递 (立即数)

```
CALL @42:forward, bind={ "$batch_size": "#64", "$lr": "#0.001" }
→ B 看到 $batch_size 绑定到标量值 64
→ 不涉及 tensor 路径解析
→ 类似 C 的 int 参数
```

### 4.3 按值传递 (tensor 深拷贝)

```
CALL @42:forward, bind={ "$input": "heap:batch" },
     flags={ "copy": true }
→ VM 在执行 CALL 前，深拷贝 heap:batch
→ 新 tensor 存于 /vproc/42/stack/batch_copy
→ $input 绑定到 stack:batch_copy
→ 开销大，仅当被调用者会修改数据时才需要
```

### 4.4 跨设备参数

```
调用者 A 的 $input 在 GPU0 (机器 n1)
被调用者 B 的默认设备是 GPU1 (机器 n2)

CALL 时的处理:
  1. VM 检测设备不匹配
  2. 选项 a: 自动 TENSOR_MOVE (迁移到 GPU1)
  3. 选项 b: B 的 executor 远程读取 (RMA, 类似 GPUDirect RDMA)
  4. 选项 c: 报错 (需要显式 TENSOR_MOVE)
  
  默认策略: 选项 a (自动迁移)，但发出 warning
  生产环境: 推荐显式管理设备位置
```

## 5. 返回值语义

### 5.1 RETURN 指令

```json
{
  "op": "RETURN",
  "values": {
    "$output": "stack:logits",
    "$loss": "stack:loss_val"
  }
}
```

### 5.2 返回值传递

```
被调用者 B 的 RETURN:
  
  情况 1: 返回的是 B 的 stack tensor
    → 调用者 A 需要该 tensor
    → 如果 A 和 B 在同一机器:
        refcount++，A 直接通过 shm 读取
    → 如果 A 和 B 在不同机器:
        需要迁移 tensor 到 A 的机器
    
  情况 2: 返回的是 B 的 heap tensor
    → A 获得对该 heap tensor 的引用
    → refcount++

  情况 3: 返回的是标量
    → 直接写入 A 的 context/registers

  情况 4: 返回路径是 PROMOTE
    → 被调用者的 stack tensor 升级为 heap tensor
    → 类似: 局部变量的地址被返回给调用者 (需要提升到堆)
```

### 5.3 多返回值

```
传统 C 语言只能返回一个值 (通过 rax)。
x86-64 ABI 可以返回两个值 (rax + rdx)。
元程可以返回任意数量:

RETURN { "$logits": "stack:logits",
         "$loss": "#0.023",
         "$accuracy": "#0.95" }

调用者 A 的 bind 指定了 return.slot:
  return.slot = "$logits"  → 只接收 $logits

如果要接收多个返回值，用 return.bind:
  return: {
    bind: { "$logits": "stack:result_logits",
            "$loss": "metrics/loss" }
  }
```

## 6. 调用栈管理

### 6.1 嵌套调用

```
train_epoch
  ├─ CALL dataloader:next_batch    (深度 1)
  │    └─ CALL preprocessor:normalize (深度 2)
  │         └─ CALL device:gpu_upload  (深度 3)
  │
  └─ CALL trainer:step              (深度 1)

最大调用深度: 建议 ≤ 32 (可配置)
超深度 → error (类似 stack overflow)
```

### 6.2 尾调用优化

```
类似 x86 的尾调用优化 (TCO):

code/forward:
  TENSOR_OP matmul ... → stack:hidden
  TENSOR_OP relu   ... → stack:output
  CALL @logger:log, bind={ "$data": "stack:output" }
  ← 这是最后一条指令，且没有后续代码

VM 可优化: 不 push call_stack，直接 JMP 到 logger 的代码。
           logger RETURN 时直接回到 forward 的调用者。

类似:
  function foo() { return bar(); }
  → 编译为 jmp bar 而非 call bar; ret
```

### 6.3 异常展开 (Stack Unwinding)

```
调用链: A → B → C → D (error!)

类似 C++ 异常展开:
  1. D 进入 error 状态
  2. VM 检查 D 的 exception_frame: 没有 error_handler
  3. VM 沿调用栈向上:
     C 的 exception_frame: 没有 error_handler
     B 的 exception_frame: 有 error_handler="on_error"
  4. VM 恢复 B 的现场
  5. 展开期间清理 C 和 D 的 stack tensor
  6. B 执行 on_error 代码块
```

## 7. 跨元程通信模式 (IPC)

元程间的通信全通过 KVSpace 中的共享路径实现，无需消息队列或 socket。

### 7.1 生产者-消费者 (通过 /tensor/)

```
生产者元程 (dataloader):              消费者元程 (trainer):
───────────────────────               ──────────────────────

TENSOR_NEW heap:batch, f32, [64,784]
TENSOR_LOAD data → heap:batch
SET /tensor/batch_queue/latest = {
  "tensor": "/vproc/dataloader/heap/batch",
  "version": 128
}                                     WATCH /tensor/batch_queue/latest
                                      → 触发 callback code="on_new_batch"
                                      
                                      on_new_batch:
                                        GET /tensor/batch_queue/latest
                                        CALL @trainer:forward,
                                          bind={ "$input": "stack:batch" }
                                        从 dataloader 的 heap 读 batch 数据
```

### 7.2 客户端-服务器 (通过同步 CALL)

```
客户端元程 (infer_api):               服务器元程 (model_server):
───────────────────────               ─────────────────────────

CALL @model_server:predict,
  bind={ "$input": "stack:query" }    收到 CALL 请求
  return.slot="$result"               │
                                      predict:
等 RETURN...                            TENSOR_OP matmul ... → $output
                                        RETURN $output
                                      
收到 $result → 返回给 API 调用者
```

### 7.3 广播-聚合 (通过 BARRIER)

```
元程 A, B, C, D 并行执行 rollout:

各自执行 code/rollout:
  ... 环境交互 ...
  TENSOR_OP ... → stack:trajectory
  BARRIER "rollout_done", count=4
  
  ← 所有 4 个元程都到达后，barrier 解除

元程 E (trainer):
  WATCH /vproc/A/status.state = "ready"
  WATCH /vproc/B/status.state = "ready"
  ...
  ← 所有 rollout 元程都返回 ready
  CALL @A:get_trajectory ...
  CALL @B:get_trajectory ...
  聚合所有 trajectory → 训练
```

### 7.4 管道模式 (流水线)

```
stage_1 元程:    preprocess → heap:stage1_out
stage_2 元程:    WATCH heap:stage1_out → forward → heap:stage2_out
stage_3 元程:    WATCH heap:stage2_out → backward → heap:gradients

管道深度 = 3。各阶段可并行处理不同 batch:
  t=0: stage1(batch0)
  t=1: stage1(batch1), stage2(batch0)
  t=2: stage1(batch2), stage2(batch1), stage3(batch0)

类似 CPU 流水线: 取指|译码|执行 重叠
```

## 8. 所有权与生命周期管理

### 8.1 返回值所有权

```
被调用者 B 返回的 tensor 所有权转移规则:

情况 A: 返回 B 的 stack tensor
  → 如果 PROMOTE → 升级为 B 的 heap tensor，refcount=2 (A+B)
  → 否则 → 深拷贝到 A 的 heap/stack
  → B 的 stack 在代码块结束时正常清理

情况 B: 返回 B 的 heap tensor  
  → A 获得引用，refcount++
  → B 仍持有引用 (refcount 至少为 2)

情况 C: 返回 /tensor/ 全局路径
  → A 获得引用，refcount++
  → 无额外所有权转移

情况 D: 返回标量立即数
  → 无所有权问题
```

### 8.2 调用期间的数据保护

```
调用者 A CALL B，传递了 A 的 tensor 引用:

B 执行期间，A 不能修改这些 tensor:
  - A 进入 wait 状态，不会执行
  - 如果 A 是异步调用且继续执行 → 需要在 tensor 上获得读锁
  
B 执行完毕后:
  - B 的 stack tensor 被清理
  - B 的 heap tensor 保持 (refcount 管理)
  - A 恢复执行
```

## 9. 错误传播

### 9.1 被调用者错误

```
B 执行 code/forward 时在 op_idx=3 失败:

错误传播:
  1. B 进入 state = "error"
  2. VM 检查 B 的 exception_frame.retry
  3. 重试耗尽后 → 错误沿调用栈向上传播到 A
  4. A 的 CALL 返回 error
  5. A 进入 error 状态或执行 on_error 代码块

错误信息包含:
  {
    "source_vpid": 42,
    "source_code": "forward",
    "source_op_idx": 3,
    "error_code": "GPU_OOM",
    "propagation_path": [42, 1]   // 错误传播链
  }
```

### 9.2 超时处理

```
CALL @slow_worker:process, bind={...}, return={ timeout_ms: 5000 }

超时后:
  1. VM 标记 B 的执行已超时
  2. 发送 CANCEL 给 B 的 executor
  3. A 收到 timeout error
  4. A 可决定重试、跳过、或降级
```

## 10. 与传统 ABI 的精妙对应

```
x86-64 CALL func(arg1, arg2)        元程 CALL @target:code, bind={...}
────────────────────────────────    ────────────────────────────────
mov rdi, arg1                      bind={ "$p1": path }
mov rsi, arg2                      bind={ "$p2": path }
call func                          CALL instruction → VM scheduling
push rbp                           追加 stack_frames
mov rbp, rsp                       当前栈帧为活动帧
... function body ...              ... 执行 code/ 中的 seq ...
mov rax, result                    RETURN { "$output": path }
leave                              弹出 stack_frames, 清理 stack tensor
ret                                恢复 caller 的 PC

cdecl (caller cleans stack)        VM 自动管理 (类似 stdcall)
fastcall (reg params)              bind 传递路径 (类似引用传递)
syscall (int 0x80)                 WATCH + CALL 的组合 (事件触发)
```

## 11. MABI 定义总结

```
┌─ 元程应用程序二进制接口 (MABI) v1 ────────────────────────────┐
│                                                                │
│  参数传递:                                                      │
│    - tensor:     bind={ "$name": path }     (引用传递, 默认)    │
│    - scalar:     bind={ "$name": "#value" } (值传递)            │
│    - deep copy:  flags={ "copy": true }     (显式深拷贝)        │
│                                                                │
│  返回值:                                                        │
│    - 单个:      return.slot = "$name"                           │
│    - 多个:      return.bind = { "$a": path, "$b": path }       │
│    - 无返回:    无 return 字段                                   │
│                                                                │
│  寄存器约定:                                                     │
│    caller-saved:   $temp0-9                                    │
│    callee-saved:   $model, $state, $config                     │
│    param slots:    $input (只读), $output (只写)                │
│                                                                │
│  栈管理:                                                        │
│    - CALL 时 push context/call_stack                           │
│    - RETURN 时 pop, 清理本地 stack tensor                       │
│    - 最大深度: 32 (可配置)                                      │
│                                                                │
│  错误处理:                                                      │
│    - 沿调用栈向上传播                                           │
│    - 每个栈帧可设 exception_frame.on_error                      │
│    - retry_count 独立可配                                       │
│                                                                │
│  所有权:                                                        │
│    - pass-by-ref 不转移所有权                                    │
│    - RETURN 的 heap tensor → refcount++                        │
│    - RETURN 的 stack tensor → PROMOTE 或深拷贝                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```
