# 元程内存空间 — KV 寻址与多级存储

> 传统进程用 32/64 位整数寻址，MMU 将虚拟地址映射到物理地址。
> 元程用 KV 路径寻址，路径解析器将 KV 路径映射到实际数据位置 (GPU shm / CPU RAM / Redis)。

## 1. 从整数寻址到 KV 寻址

```
传统进程:
  CPU 发出: mov rax, [0x7fff12340000]
  MMU 查页表: 0x7fff12340000 → 物理页帧 0xABCD000
  → 读物理内存

元程:
  VM 执行: TENSOR_LOAD heap:weights → $slot
  路径解析器: heap:weights → /vproc/42/heap/weights
  → GET Redis → {shm:"/deepx_t_xxx", device:"gpu0"}
  → executor shm_open → 读 GPU 内存
```

**关键洞察：** KV 路径是一种**符号地址**。它不像整数地址那样映射到连续的字节偏移，而是映射到**命名的、有类型的、分布式的**数据对象。这是一种比虚拟内存更高级的抽象——不是"第 N 个字节"，而是"名为 weights 的 tensor"。

## 2. 元程地址空间布局

```
/vproc/<vpid>/                     ← 元程的"虚拟地址空间"根
│
├── code/                          代码段 (.text)
│   ├── init                       $0x0000  (符号地址)
│   ├── forward                    $0x0001
│   └── backward                   $0x0002
│
├── method/                        方法区 (符号表 + 能力注册)
│   ├── matmul                     元程能执行的算子
│   └── add
│
├── heap/                          堆段 (.data + .bss)
│   ├── weights                    全局可共享, refcount 管理
│   ├── bias
│   └── embedding
│
├── stack/                         栈段 (.stack)
│   ├── hidden                     临时, 代码块作用域
│   ├── output                     带 TTL, 自动回收
│   └── grad_hidden
│
├── context/                       元程控制块 (PCB等价)
│   ├── registers                  命名槽位状态
│   ├── stack_frame                当前栈帧
│   └── watch_list                 事件监听列表
│
├── info                           元程元信息
├── status                         运行状态 (PC等价在此)
├── config/                        配置段
├── metrics/                       性能计数器
└── log/                           日志缓冲区
```

## 3. 寻址模式

| 模式 | 语法 | 解析 | 类比 x86 |
|------|------|------|---------|
| 直接寻址 (堆) | `heap:weights` | `/vproc/self/heap/weights` | `[absolute_addr]` |
| 直接寻址 (栈) | `stack:hidden` | `/vproc/self/stack/hidden` | `[rbp-8]` |
| 寄存器寻址 | `$input` | 调用者绑定的路径 | `rax` |
| 绝对寻址 | `/tensor/model/w` | 直接使用 | `[0xdeadbeef]` |
| 跨元程寻址 | `@42:weights` | `/vproc/42/heap/weights` | 跨进程共享内存 |
| 立即数 | `#3.14` | 标量常量 | `$3.14` |
| 间接寻址 | `heap:layer_$i` | 模板展开后解析 | `[rbx+rax*4]` |

## 4. 路径解析器 (Path Resolver)

类比 MMU。将 KV 路径翻译为数据所在位置：

```
┌─ 路径解析器 ─────────────────────────────────────────────────┐
│                                                               │
│  输入: heap:weights                                           │
│                                                               │
│  Step 1: 前缀展开                                             │
│    heap:weights → /vproc/42/heap/weights                      │
│                                                               │
│  Step 2: 本地缓存查询 (路径缓存, 类似 TLB)                      │
│    命中 → 直接返回 {shm, device, ptr}                          │
│    未命中 → 继续                                               │
│                                                               │
│  Step 3: Redis 查询 (类似页表遍历)                              │
│    HGETALL /vproc/42/heap/weights                             │
│    → {dtype:"f32", shape:[1024,512],                          │
│       shm:"/deepx_t_xxx", device:"gpu0", version:5}           │
│                                                               │
│  Step 4: shm 解析 (类似物理地址计算)                            │
│    shm_open("/deepx_t_xxx") → fd                              │
│    mmap(fd) → CPU virtual ptr                                 │
│    newBufferWithBytesNoCopy → GPU ptr                         │
│                                                               │
│  Step 5: 缓存结果到路径缓存                                    │
│                                                               │
│  输出: {gpu_ptr, byte_size, dtype, version}                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 5. 多级存储层次

传统计算机的内存层次：寄存器 → L1 → L2 → L3 → RAM → SSD。
元程的存储层次：

```
┌────────────────────────────────────────────────────────────┐
│ Level 0: 命名槽位 ($input, $temp0)                          │
│  类比: CPU 寄存器                                          │
│  延迟: 0 (直接引用已有 tensor)                               │
│  容量: ~10 个活跃引用                                        │
│  生命周期: 单条指令                                          │
├────────────────────────────────────────────────────────────┤
│ Level 1: executor 本地路径缓存                               │
│  类比: L1/L2 Cache + TLB                                   │
│  延迟: ~ns (内存查表)                                        │
│  容量: ~100 个最近使用的 tensor 映射                         │
│  一致性: 需失效机制 (version check)                          │
├────────────────────────────────────────────────────────────┤
│ Level 2: GPU 显存 (MTLBuffer / CUDA ptr)                   │
│  类比: L3 Cache + RAM                                      │
│  延迟: ~μs (GPU kernel 内访问)                               │
│  容量: ~GB                                                  │
│  一致性: MTLStorageModeShared / CUDA IPC                    │
├────────────────────────────────────────────────────────────┤
│ Level 3: CPU 内存 (同一机器)                                 │
│  类比: RAM (同 NUMA 节点)                                    │
│  延迟: ~μs (PCIe/统一内存带宽)                                │
│  容量: ~GB                                                  │
│  迁移: TENSOR_MOVE gpu→cpu                                  │
├────────────────────────────────────────────────────────────┤
│ Level 4: 远端机器 GPU/CPU                                   │
│  类比: 远端 NUMA 节点 / 网络存储                             │
│  延迟: ~ms (网络传输)                                        │
│  容量: 集群总显存                                            │
│  迁移: 显式复制或 RMA                                        │
├────────────────────────────────────────────────────────────┤
│ Level 5: KVSpace (Redis) — 仅元数据                         │
│  类比: 页表 + inode                                          │
│  延迟: ~μs (Redis 查询)                                      │
│  存储: 描述信息，不含实际数据                                  │
├────────────────────────────────────────────────────────────┤
│ Level 6: 持久化存储 (checkpoint 文件)                        │
│  类比: SSD / swap                                          │
│  延迟: ~ms-s                                                │
│  用途: 故障恢复、模型版本管理                                  │
└────────────────────────────────────────────────────────────┘
```

## 6. 页表 vs 路径表

```
传统 OS 页表:
  虚拟地址 → 物理地址
  4KB 粒度
  多级页表 (PGD→PUD→PMD→PTE)
  TLB 缓存

元程路径表:
  KV 路径 → {device, shm, ptr}
  tensor 粒度 (任意大小)
  多级路径解析 (前缀→Redis→shm)
  路径缓存 (类似 TLB)
```

**关键差异：** OS 页表按 4KB 切割所有内存，对 tensor 这种大对象效率低。
元程路径表以 tensor 为最小单元，一次解析即获得完整数据引用。

## 7. 缺页异常 → 缺张量异常

```
OS 缺页异常:
  CPU 访问不在 RAM 中的页
  → 页故障处理程序
  → 从 swap 换入
  → 更新页表
  → 重试指令

元程缺张量异常:
  executor 访问不在本地 GPU 中的 tensor
  → 路径解析器发现 device != 本地
  
  处理策略:
  a) 自动迁移: TENSOR_MOVE 到本地设备
  b) 远程访问: 跨机器发送 (类似 RMA)
  c) swap-in: 从 checkpoint 文件加载
  
  → 更新路径缓存
  → 重试算子
```

## 8. 内存分配器

```
传统 malloc():
  从堆中分配连续字节，返回指针
  用 free() 释放
  brk() / mmap() 向 OS 申请内存

元程 TENSOR_NEW:
  从全局堆中分配命名 tensor
  VM 通知 heapmem 创建 shm + GPU buffer
  refcount = 1
  写入 /vproc/<vpid>/heap/<name>
  
元程 TENSOR_DEL:
  refcount-- (Redis DECR)
  若 refcount = 0:
    heapmem 回收 shm (shm_unlink)
    清理 /vproc/<vpid>/heap/<name>
    清理 /tensor/<name>/
  
  若 refcount > 0:
    仅 dec，不回收 (其他元程仍在使用)
```

## 9. 内存碎片与压缩

```
OS: 内存碎片 → 紧凑 (compaction) / 垃圾回收
元程: tensor 碎片化 → tensor 合并/拆分

GPU 显存碎片:
  - heapmem 使用 buddy allocator 管理 shm 池
  - 大 tensor 分配失败 → 触发 compaction:
    TENSOR_MOVE 重组空闲块
  - 跨设备卸载: 冷 tensor → CPU, 热 tensor → GPU
```
