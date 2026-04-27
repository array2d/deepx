# POSIX Shared Memory 跨进程 MTLBuffer 方案

> 目标：验证 macOS Metal 下，通过 POSIX 共享内存 + `newBufferWithBytesNoCopy` 实现跨进程 tensor 共享的可行性。

## 1. 背景

deepx 在 CUDA 侧采用双进程架构：

- **heapmem-cuda**：管理堆 tensor 生命周期，通过 `cudaIpcGetMemHandle` 导出 GPU 显存句柄到 Redis
- **op-cuda**：计算进程，通过 `cudaIpcOpenMemHandle` 导入 GPU 显存，执行算子

Metal 没有 CUDA IPC 等价物。本方案探索用 POSIX 共享内存 (`shm_open` + `mmap`) + `-[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]` 替代。

## 2. 核心假设

在 Apple Silicon 统一内存架构 (UMA) 下：

1. POSIX shm 分配的物理内存与 GPU 可访问的物理内存是同一块
2. `newBufferWithBytesNoCopy` 接受任何页对齐的 CPU 虚拟地址，Metal 驱动将其绑定为 GPU 可访问
3. 两个独立进程各自映射同一 shm 并创建 MTLBuffer，它们的 GPU 操作作用于同一物理页
4. GPU L2 cache 在 `[commandBuffer waitUntilCompleted]` 后对同一物理内存的后续访问者可见

```
┌──────────────────────────────────────────────────────┐
│                  POSIX 共享内存 (/deepx_shm_test)      │
│                                                        │
│   ┌──────────────────────────────────────────────┐    │
│   │  物理页 A │ 物理页 B │ 物理页 C │ ...         │    │
│   └──────────────────────────────────────────────┘    │
│         ▲                                 ▲            │
│   mmap  │                            mmap │            │
│   ┌─────┴──────────┐         ┌───────────┴───────┐    │
│   │ Process A      │         │ Process B         │    │
│   │ heapmem-mps    │         │ op-mps            │    │
│   │                │  Redis  │                   │    │
│   │ MTLBuffer ─────┼────shm──┼─── MTLBuffer      │    │
│   │ (no-copy)      │  name   │   (no-copy)       │    │
│   └────────────────┘         └───────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## 3. 验证方案

### 3.1 Stage 1 — 单进程基线

验证 `newBufferWithBytesNoCopy` + POSIX shm 的 GPU 可访问性：

```
shm_open → ftruncate → mmap → newBufferWithBytesNoCopy
→ Metal kernel dispatch (A + B → C)
→ waitUntilCompleted
→ CPU 读取 C 验证结果
```

### 3.2 Stage 2 — fork 跨进程

```
Parent:  shm_open + ftruncate + mmap
         → newBufferWithBytesNoCopy → CPU write A, B
         → Metal kernel: C = A + B
         → waitUntilCompleted
         → POSIX sem_post (通知 child)

Child:   sem_wait (等待 parent)
         → shm_open + mmap (同一 shm name)
         → CPU 读取 C
         → 验证 C[i] == A[i] + B[i]
```

### 3.3 Stage 3 — 独立进程

两个独立编译的二进制，通过共享 shm name + semaphore name 协作。

## 4. 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| GPU MMU 隔离导致 GPU 无法访问 shm 物理页 | 中 | 致命 | Stage 1 暴露 |
| GPU cache 跨进程不可见 | 中 | 高 | waitUntilCompleted + semaphore |
| newBufferWithBytesNoCopy 拒绝 shm 指针 | 低 | 中 | 换 IOSurface |
| 页对齐不满足 | 低 | 低 | mmap + ftruncate 保证 |
| MTLBuffer 各自生命周期冲突 | 低 | 中 | semaphore 串行化 GPU 操作 |

## 5. 目录

```
executor/op-mem-mps/test/shm/
├── CMakeLists.txt
├── test_shm_mtl_baseline.mm      # Stage 1
├── test_shm_cross_fork.mm        # Stage 2
└── test_shm_cross_proc.mm        # Stage 3
```

## 6. 成功标准

- [ ] Stage 1: GPU kernel 读写 POSIX shm 映射的 MTLBuffer 成功，CPU 验证通过
- [ ] Stage 2: fork 子进程通过 shm 读取父进程 GPU 计算结果正确
- [ ] Stage 3: 两个独立进程通过命名 shm 共享 tensor 数据成功

## 7. 后续架构

验证通过后，heapmem-mps:

```
heapmem-mps 进程:
  - shm_open 创建/获取 tensor 共享内存
  - Redis 管理 tensor name → shm name 映射
  - 引用计数和 GC
  - 响应 Redis lifecycle 指令

op-mps 进程:
  - Redis 获取 tensor → shm name
  - shm_open + mmap 映射 → newBufferWithBytesNoCopy
  - GPU kernel 执行
  - POSIX semaphore 跨进程同步
```
