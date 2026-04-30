# heap-cuda

> Linux CUDA GPU 显存的 heap-plat 实现。待开发。
>
> **heap-cuda 的进程维持着 deepx 元程的堆在 CUDA 设备平台的高可用。**

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| 设备 | NVIDIA GPU (CUDA) |
| 内存模型 | 独立显存 (VRAM)，需 cudaMemcpy 传输 |
| 分配方式 | cudaMalloc + POSIX shm (寄存器映射) |
| shm 命名 | `/deepx_t_<8位随机hex>` |
| GPU 访问 | CUDA kernel 直接通过 device ptr 访问 |

## 2. 设计要点

### 与 Metal 的关键差异

Metal (统一内存):
```
mmap ptr ←→ GPU 直接访问 (零拷贝)
```

CUDA (独立显存):
```
shm (CPU 可访问) ←→ cudaMemcpy ←→ VRAM (GPU 可访问)
```

需要一个额外的共享内存段来记录 VRAM 指针，供 op-cuda 获取。

### 方案

```
1. newtensor:
   a) 分配 shm (CPU 侧, 用于跨进程交换元信息)
   b) cudaMalloc(&dptr, byte_size) → VRAM
   c) 将 dptr 写入 shm 头部 (如 offset 0, sizeof(void*) 字节)
   d) shm_name 和 dptr 写入 Redis 元信息

2. tensor 元信息 (多一个 vram_ptr):
   {
       "dtype": "f32",
       "shape": [1024, 512],
       "byte_size": 2097152,
       "device": "gpu0",
       "address": {
           "type": "cuda",
           "shm_name": "/deepx_t_abc123",
           "vram_ptr": "0x7f...",       // cudaMalloc 返回的 device ptr
           "cuda_ctx": "<context_id>"   // CUDA context (多 GPU 场景)
       },
       "ctime": 1714000000,
       "version": 1
   }
```

### 多 GPU 场景

每张 GPU 独立一个实例:

```
heap-cuda:0 → gpu0 → 管理 /dev/nvidia0
heap-cuda:1 → gpu1 → 管理 /dev/nvidia1
```

CUDA context 与实例一一绑定。clonetensor 跨卡时:
```c
// GPU0 → GPU1 的 clone (通过 P2P 或中转)
cudaMemcpyPeer(dst_ptr_gpu1, 1, src_ptr_gpu0, 0, byte_size);
```

## 3. 待开发

| 任务 | 说明 |
|------|------|
| CUDA 设备初始化 | cudaSetDevice, context 管理 |
| cudaMalloc / cudaFree | VRAM 分配释放 |
| CPU↔GPU 数据桥 | shm ↔ VRAM 的 memcpy 封装 |
| P2P 传输 (clonetensor) | cudaMemcpyPeer 跨 GPU |
| 进程注册 | /sys/heap-plat/cuda:0 |

## 4. 依赖

- CUDA Toolkit (libcuda)
- hiredis

## 5. 开发量

~500 行 C++/CUDA C。
