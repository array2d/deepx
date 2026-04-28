# op-cpu

> 纯 CPU 计算的 op-plat 实现。待开发。作为最小化参考实现和测试基准。

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| 设备 | CPU (无 GPU) |
| 后端 | C++ 循环 (可集成 BLAS) |
| 内存 | 系统虚拟内存 |
| 算子 | 基础: elementwise, matmul (BLAS), reduce |

## 2. 设计要点

最简实现，不依赖任何 GPU API:

```
1. 消费指令: RPOP cmd:op-cpu:0
2. 映射数据: shm_open → mmap → CPU ptr
3. 执行计算: C++ 循环 (可能调用 OpenBLAS)
4. 写回结果: 直接写入 mmap 区域
5. 通知完成: LPUSH done:<vtid>
```

**性能优化 (可选):**
- OpenMP 多线程并行
- SIMD 向量化 (编译优化 -O3 -march=native)
- 大页内存 (MAP_HUGETLB)

## 3. 计划算子

| 类别 | 算子 | 实现方式 |
|------|------|---------|
| elementwise | add, sub, mul, div | 简单 C++ 循环 |
| activation | relu, sigmoid, tanh, gelu | C++ 数学库 |
| matmul | matmul | OpenBLAS SGEMM |
| reduce | sum, mean, max | C++ 循环 / BLAS |
| changeshape | reshape | 零拷贝 (共享 shm, 改 shape 元信息) |
| changeshape | transpose | 索引重排循环 |
| init | zeros, ones, arange | memset / 简单循环 |

## 4. Tensor 访问

```
GET /vthread/1/a → tensor 元信息
shm_open(shm_name) + mmap → float* data_ptr
直接读写 data_ptr (CPU 零开销)
```

## 5. 算子注册

```
/op/op-cpu/list = [
    "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu",
    "matmul",
    "sum", "mean", "max",
    "reshape", "transpose",
    "zeros", "ones", "arange"
]
```

## 6. 进程注册

```
/sys/op-plat/cpu:0 = {"program":"op-cpu", "device":"cpu", "status":"running", "load":0.0, "pid":<pid>}
```

## 7. 依赖

| 依赖 | 说明 |
|------|------|
| hiredis | Redis 客户端 |
| OpenBLAS (可选) | 高性能矩阵乘法 |
| OpenMP (可选) | 多线程并行 |

## 8. 开发量

~500 行 C++ (基础实现) + 整合 BLAS。
