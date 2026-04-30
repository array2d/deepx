# dxlang — 已迁移至 deepx-core

**dxlang 的所有类型系统代码已迁移至 `executor/deepx-core/`。**

## 迁移内容

| 原路径 | 新路径 |
|--------|--------|
| `src/deepx/dtype/precision.hpp` | `deepx-core/include/deepx/precision.hpp` |
| `src/deepx/dtype/data_category.hpp` | `deepx-core/include/deepx/data_category.hpp` |
| `src/deepx/dtype/typespec.hpp` | `deepx-core/include/deepx/typespec.hpp` |
| `src/stdutil/*` | `deepx-core/include/stdutil/*` + `deepx-core/src/stdutil/*` |

## 使用方式

所有 executor 现在通过 deepx-core 获取类型系统：

```cmake
add_subdirectory(../deepx-core deepx-core)
target_link_libraries(your_target deepx_core)
```

```cpp
#include "deepx/precision.hpp"
#include "deepx/typespec.hpp"
```

## 设计理念

dxlang 的设计理念（语言先于执行、IR-first、稳定契约、执行解耦、最小依赖）仍然是 deepx-core 的指导原则，详见 `doc/deepx-core/README.md`。
