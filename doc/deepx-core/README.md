# deepx-core

> deepx-core 是 deepx 所有 C++ 执行器的统一平台无关公共库。
> 提供类型系统(dtype)、张量类型(Shape/Tensor)、共享内存(shmem)、注册接口(Registry)与工具类(stdutil)。
> 不绑定任何特定硬件后端（Metal/CUDA/ROCm）。

## 定位

- **面向**: 所有 executor（exop-cpu, op-metal, op-cuda, heap-metal, heap-cpu, io-metal）
- **提供**: 统一类型系统、张量数据结构、POSIX 共享内存管理、抽象注册接口
- **不提供**: 硬件 kernel（Metal/CUDA/CPU SIMD）、调度策略、通信协议

## 核心模块

### dtype — 类型系统

| 头文件 | 内容 |
|--------|------|
| `deepx/precision.hpp` | Precision 位图枚举：Float64~Float4E2M1, Int64~Int4, Bool, String；precision_bits / from_string / to_string |
| `deepx/data_category.hpp` | DataCategory 位图枚举：Var, Vector, Tensor, ListTensor |
| `deepx/typespec.hpp` | TypeSpec 联合体：DataCategory + Precision 组合，含 match / autodtype |

### tensor — 张量类型

| 头文件 | 内容 |
|--------|------|
| `deepx/shape.hpp` | Shape 结构体：dtype, shape, strides, size, bytes()；含 YAML 序列化 |
| `deepx/tensor_base.hpp` | TensorBase：轻量基类，仅持有 Shape |
| `deepx/tensor.hpp` | Tensor\<T\> 模板：data 指针 + 内存管理函数（RAII） |
| `deepx/shape_changeshape.hpp` | transpose/concat/broadcast/indexselect/repeat 形状计算 |
| `deepx/shape_matmul.hpp` | 矩阵乘法形状计算 |
| `deepx/shape_reduce.hpp` | reduce 归约形状计算 |
| `deepx/shape_tensorinit.hpp` | 张量初始化形状计算 |
| `deepx/vector_combination.hpp` | 向量组合工具 |

### tensorfunc — 算子 Dispatch 接口

| 头文件 | 内容 |
|--------|------|
| `tensorfunc/authors.hpp` | Author 标签（default_, miaobyte, cblas, cublas） |
| `tensorfunc/elementwise.hpp` | Dispatcher：add/sub/mul/div/pow/sqrt/log/exp/sin/cos/tan/max/min/equal/less/greater/invert/switch |
| `tensorfunc/matmul.hpp` | Dispatcher：matmul |
| `tensorfunc/init.hpp` | Dispatcher：constant/arange/dropout/uniform/normal |
| `tensorfunc/reduce.hpp` | Dispatcher：reducemax/reducemin/sum/prod |
| `tensorfunc/changeshape.hpp` | Dispatcher：reshape/transpose/concat/broadcastTo/indexselect/repeat/repeat_interleave |
| `tensorfunc/io.hpp` | Dispatcher：print |
| `tensorfunc/tensorlife.hpp` | 张量生命周期管理 |

### tf — TF 框架

| 头文件 | 内容 |
|--------|------|
| `tf/tf.hpp` | TF 类（name/tftype/args/returns/metadata）；Param；TFMetadata；OpResp |
| `tf/tffactory.hpp` | TfFactory → TFFamily → TFAuthor 多层注册与类型匹配调度 |

### mem — 内存管理

| 头文件 | 内容 |
|--------|------|
| `mem/mem.hpp` | MemBase：args(map) + tensor 存储(map)；addarg/getarg/addtensor/gettensor |

### shmem — 共享内存

| 头文件 | 内容 |
|--------|------|
| `deepx/shm_tensor.h` | POSIX shm 张量：shm_tensor_create / open / close / unlink |

### registry — 注册接口

| 头文件 | 内容 |
|--------|------|
| `deepx/registry.h` | 抽象 Registry 接口：create_or_get / ref_inc / ref_dec / get_meta |

### stdutil — 工具类

| 头文件 | 内容 |
|--------|------|
| `stdutil/num.hpp` | is_integer / is_float / is_positive_integer |
| `stdutil/string.hpp` | trim / trimspace / escape_markdown |
| `stdutil/error.hpp` | NotImplementError / UnsupportedOperationException / TensorShapeError |
| `stdutil/fs.hpp` | 文件系统工具（filename 等） |
| `stdutil/time.hpp` | 时间工具 |
| `stdutil/vector.hpp` | 向量工具 |
| `stdutil/print.hpp` | 打印工具 |

## 构建

deepx-core 编译为静态库 `libdeepx_core.a`：

```bash
cd executor/deepx-core
cmake -S . -B build
cmake --build build -j
```

## 依赖关系

```
deepx-core (libdeepx_core.a)
├── 外部依赖: yaml-cpp (Shape YAML 序列化)
└── 被依赖:
    ├── exop-cpu     (CPU 算子引擎)
    ├── op-metal     (Metal GPU 算子引擎，通过 common-metal HAL)
    ├── op-cuda      (CUDA 算子引擎，规划中)
    ├── heap-metal   (Metal 堆管理，通过 common-metal HAL)
    ├── heap-cpu     (CPU 堆管理)
    └── heap-cuda    (CUDA 堆管理，规划中)
```

## 与其他组件的关系

```
deepx-core (dtype + tensor + shmem + registry + stdutil)
    │
    ├── common-metal (HAL only): metal_device，额外依赖 Metal.framework
    │       │
    │       ├── op-metal
    │       └── heap-metal
    │
    ├── exop-cpu (直接依赖，无 HAL 层)
    ├── heap-cpu (直接依赖，无 HAL 层)
    └── op-cuda (直接依赖，无 HAL 层，规划中)
```

## 迁移说明

deepx-core 整合了以下三个原有库的平台无关部分（全部迁移完成，旧目录均已清理）：

| 原库 | 迁移内容 | 状态 |
|------|---------|------|
| `dxlang/` | dtype (precision/data_category/typespec), stdutil (7+3) | 源码已删除，仅保留 README |
| `common-metal/` | shm_tensor, registry | 已删除，仅保留 Metal HAL |
| `old-cppcommon/` | 全部 29 个文件：核心类型(5) + shape辅助(11) + tensorfunc(8) + tf(4) + mem(1) | 目录已删除（rm -rf） |

## 设计原则

1. **平台无关**: 不依赖任何特定操作系统/硬件 API（POSIX shm 除外，为可选项）
2. **最小依赖**: 仅依赖 STL + yaml-cpp
3. **静态链接**: `.a` 静态库，避免运行时库查找问题
4. **扁平头文件**: 所有公共头文件位于 `include/deepx/` 单层目录，无子目录嵌套
5. **头文件与实现分离**: `include/` 为公共 API，`src/` 为编译单元
