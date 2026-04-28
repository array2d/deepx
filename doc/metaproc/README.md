# 元程 (Metaproc) 体系设计

> 元程是 deepx 的一等计算实体。它将传统 OS 进程中"由内核管理的一切"
> 重新表达为"由 KVSpace 管理的元数据 + 由 executor 执行的分布式计算"。

## 与传统进程的对应

| OS 进程概念 | 元程对应 | 实现 |
|------------|---------|------|
| 进程控制块 (PCB) | 元程上下文 (MCB) | `/vproc/<vpid>/status` + `context/` 子树 |
| 虚拟地址空间 | KV 寻址空间 | 路径代替整数地址 |
| 指令集 (ISA) | 元程指令集 (MISA) | dxlang 算子序列 |
| 通用寄存器 | 命名槽位 | `$input`, `$temp0`, `stack:name` |
| 程序计数器 (PC) | 当前算子索引 | `status.current_op_idx` |
| 栈指针 (SP) | 当前栈帧 | `context/stack_frame` |
| 页表 (Page Table) | 路径解析器 | `heap:name` → shm 地址 |
| 代码段 (.text) | 代码区 | `/vproc/<vpid>/code/` |
| 数据段 (.data, .bss) | 堆区 | `/vproc/<vpid>/heap/` |
| 栈段 (.stack) | 栈区 | `/vproc/<vpid>/stack/` |
| 文件描述符表 | 全局 tensor 引用 | `/tensor/<name>` |
| 内核态/用户态 | VM 调度器 / executor | 控制面 vs 数据面 |
| 中断向量表 | WATCH 回调 | 事件驱动 |
| 上下文切换 | 算子调度 | VM 路由到 executor |
| TLB | 路径缓存 | executor 本地 shm 映射缓存 |
| 缺页异常 | 堆 tensor swap-in | GPU↔CPU 数据迁移 |
| fork() | 元程克隆 | 子树复制 + tensor refcount++ |
| exec() | 代码替换 | SET code/* |
| 信号 | WATCH 事件 | 异步通知 |
| 管道/共享内存 | /tensor/ 共享 | 全局堆 tensor |

## 设计文档索引

| 文档 | 内容 |
|------|------|
| [isa.md](isa.md) | 元程指令集架构 (MISA) |
| [memory.md](memory.md) | 元程内存空间 — KV 寻址与多级存储 |
| [cache.md](cache.md) | 多级缓存一致性协议 |
| [context.md](context.md) | 元程上下文、状态机与调度 |
| [calling.md](calling.md) | 调用约定与跨元程通信 ABI |

## 核心洞察

OS 进程的三个基石：**寻址、指令、调度**。元程用 KV 寻址取代整数寻址，用 dxlang 算子取代机器指令，用 VM 调度器取代内核调度器。其余一切（堆、栈、缓存、上下文、同步）均由此三角推导而出。
