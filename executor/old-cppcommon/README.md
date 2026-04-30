# old-cppcommon

> ⚠️ 旧版 C++ 公共代码。核心类型（Shape/Tensor/TensorBase/dtype）已迁移至 `deepx-core`。

## 迁移状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `dtype.hpp` | ❌ 已删除 | 空文件，已由 `deepx-core` 的 `precision.hpp` + `data_category.hpp` + `typespec.hpp` 替代 |
| `shape.hpp` | ❌ 已删除 | 已迁移至 `deepx-core/include/deepx/shape.hpp` |
| `tensor.hpp` | ❌ 已删除 | 已迁移至 `deepx-core/include/deepx/tensor.hpp` |
| `tensorbase.hpp` | ❌ 已删除 | 已迁移至 `deepx-core/include/deepx/tensor_base.hpp` |
| `shape.cpp` | ❌ 已删除 | 已迁移至 `deepx-core/src/tensor/shape.cpp` |

## 保留内容

### tensorfunc/ — 算子接口（Dispatcher 模式）
多后端可复用的算子接口定义，被 `exop-metal` 的 `*_miaobyte.hpp` 实现所继承：
- `authors.hpp` — Author 标签（default_, miaobyte, cblas, cublas）
- `elementwise.hpp` — Elementwise 算子 Dispatcher
- `matmul.hpp` — 矩阵乘法 Dispatcher
- `init.hpp` — 初始化算子 Dispatcher
- `reduce.hpp` — 缩减算子 Dispatcher
- `changeshape.hpp` — 形状变换算子 Dispatcher
- `io.hpp` — I/O 算子 Dispatcher
- `tensorlife.hpp` — Tensor 生命周期管理接口

### tf/ — TF 框架
旧架构的算子注册与调度框架（UDP + MemBase），被 `exop-metal` 使用：
- `tf.hpp` / `tf.cpp` — TF 类（name/tftype/args/returns/metadata）
- `tffactory.hpp` / `tffactory.cpp` — TfFactory → TFFamily → TFAuthor 多层注册

### mem/ — 内存管理
- `mem.hpp` — MemBase：args(map) + tensor 存储(map)

### client/ — 旧通信方式（逐步淘汰）
- `udpserver.hpp/cpp` — UDP 服务器
- `unixsocketserver.hpp/cpp` — Unix Socket 服务器
- `worker.hpp` — 工作线程

### shape_*.hpp/cpp — Shape 辅助工具
被 `exop-metal` 的 `miaobyte` 实现引用：
- `shape_changeshape.hpp/cpp` — transpose/concat/broadcast/indexselect/repeat shape 计算
- `shape_matmul.hpp/cpp` — matmul shape 验证
- `shape_range.hpp/cpp` — range 遍历（含 OpenMP 并行）
- `shape_reduce.hpp/cpp` — reduce dims 检查与计算
- `shape_tensorinit.hpp/cpp` — fan_in/fan_out 计算
- `vector_combination.hpp/cpp` — 向量组合工具

## 依赖

所有保留代码的 `#include` 已更新为使用 `deepx-core` 的规范路径：
- `#include "deepx/tensor.hpp"` — Tensor\<T\> 模板
- `#include "deepx/shape.hpp"` — Shape 结构体
- `#include "deepx/precision.hpp"` — Precision 枚举
- `#include "deepx/typespec.hpp"` — TypeSpec 联合体

## 未来方向

`tensorfunc/` + `tf/` 框架最终将被淘汰，由各 executor 的 Redis+shm 原生实现替代（参见 `exop-cpu/src/client/main.cpp`）。
