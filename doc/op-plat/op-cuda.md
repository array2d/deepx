# op-cuda

> Linux NVIDIA GPU 的 op-plat 实现。当前最成熟的 op-plat，47 个文件。

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| GPU | NVIDIA (CUDA) |
| 后端 | CUDA C++ (nvcc) |
| 内存 | 独立显存 (VRAM) |
| 传输 | cudaMemcpy (CPU↔GPU) |
| 算子 | elementwise, changeshape, reduce, matmul, init, io 全覆盖 |

## 2. 代码位置

```
executor/op-cuda/       (47 文件, 已成熟)
├── elementwise/        add, sub, mul, div, relu, sigmoid, ...
├── changeshape/        reshape, transpose, concat, slice
├── reduce/             sum, mean, max, min
├── matmul/             matmul (cuBLAS)
├── init/               zeros, ones, arange
└── io/                 数据读写

依赖 common-cuda
```

## 3. 算子清单 (全覆盖)

| 类别 | 算子 | 实现方式 |
|------|------|---------|
| elementwise | add, sub, mul, div | CUDA kernel |
| activation | relu, sigmoid, tanh, gelu, silu | CUDA kernel |
| matmul | matmul | cuBLAS |
| reduce | sum, mean, max, min | CUDA kernel (parallel reduction) |
| changeshape | reshape, transpose, concat, slice | CUDA kernel / cudaMemcpy |
| init | zeros, ones, arange, constant | CUDA kernel |
| fused | fused_matmul_add_relu, fused_linear_norm | CUDA kernel (融合) |
| io | load, save | cudaMemcpy |

## 4. 算子注册

```
/op/op-cuda/list = [
    "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu",
    "matmul",
    "sum", "mean", "max", "min",
    "reshape", "transpose", "concat", "slice",
    "zeros", "ones", "arange", "constant",
    "fused_matmul_add_relu", "fused_linear_norm",
    "load", "save"
]

/op/op-cuda/fused_matmul_add_relu = {
    "category": "fused",
    "dtype": ["f32", "f16", "bf16"],
    "replaces": ["matmul", "add", "relu"]
}
```

## 5. VRAM 访问

与 Metal 统一内存不同，CUDA 有独立显存。tensor 元信息中需包含 VRAM 指针：

```json
{
    "dtype": "f32",
    "shape": [1024, 512],
    "byte_size": 2097152,
    "device": "gpu0",
    "address": {
        "type": "cuda",
        "shm_name": "/deepx_t_abc123",
        "vram_ptr": "0x7f1234000000",
        "cuda_ctx": "0"
    }
}
```

**VRAM 指针共享方式:**
heap-cuda 分配 VRAM 后，将 `vram_ptr` 写入 shm 头部固定 offset。
op-cuda 通过 shm_open 读取该字段获得 VRAM 指针，无需重新 cudaMalloc。

```
shm layout:
  [0..7]:   uint64 vram_ptr         ← heap-cuda 写入
  [8..15]:  uint64 cuda_context
  [16..]:   tensor 实际数据 (CPU 可见副本, 可选)
```

## 6. 多 GPU 场景

```
/sys/op-plat/cuda:0  → gpu0 → cmd:op-cuda:0
/sys/op-plat/cuda:1  → gpu1 → cmd:op-cuda:1
```

每个进程实例绑定一张 GPU (cudaSetDevice)。跨 GPU 的数据通过编译器插入的
clonetensor (heap-cuda P2P) 或 cudaMemcpyPeer 完成。

## 7. 待改造

当前 op-cuda 代码已成熟，主要改造点:

| 改造项 | 说明 |
|------|------|
| Redis 命令循环 | 替换当前通信方式为 Redis List 消费 |
| 算子注册 | 启动时 SET /op/op-cuda/list 和元数据 |
| 完成通知 | LPUSH done:<vtid> |
| 进程注册 | SET /sys/op-plat/cuda:N |
| 批量执行 | batch 指令并行 (多 CUDA stream) |

## 8. 开发量

主要是适配层 (~400 行 C++/CUDA C)，不涉及新 kernel 开发。
