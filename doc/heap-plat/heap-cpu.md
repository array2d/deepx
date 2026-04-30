# heap-cpu

> 纯 CPU 内存的 heap-plat 实现。待开发。
>
> **heap-cpu 的进程维持着 deepx 元程的堆在 CPU 设备平台的高可用。**

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| 设备 | CPU (无 GPU) |
| 内存模型 | 传统虚拟内存 |
| 分配方式 | POSIX shm + 普通 mmap/malloc |
| GPU 访问 | 不适用 |

## 2. 设计要点

最简实现，无需 GPU 相关逻辑:

```
1. newtensor:
   a) shm_open + ftruncate + mmap → CPU ptr
   b) SET Redis 元信息

2. deltensor:
   a) shm_unlink
   b) DELETE Redis key
```

无 GPU 上下文，无 VRAM 分配，无 memcpy 传输。

## 3. Tensor 元信息

```json
{
    "dtype": "f32",
    "shape": [1024, 512],
    "byte_size": 2097152,
    "device": "cpu",
    "address": {
        "type": "shm",
        "shm_name": "/deepx_t_abc123"
    }
}
```

## 4. 进程注册

```
/sys/heap-plat/cpu:0 = {"program":"heap-cpu", "device":"cpu", "status":"running", "pid":<pid>}
```

## 5. 待开发

| 任务 | 说明 |
|------|------|
| shm 分配/释放 | shm_open / shm_unlink |
| 大页支持 | mmap MAP_HUGETLB (可选) |
| NUMA 感知 | mbind (可选) |

## 6. 依赖

- hiredis
- librt (shm_open)

## 7. 开发量

~250 行 C/C++ (最简实现)。
