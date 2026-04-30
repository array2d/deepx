# C++ 核心库整合方案

## 1. 现状分析

当前 `executor/` 下存在 3 个 C++ 核心库：

```
executor/
├── dxlang/           # CMake project: deepxcore → libdeepxcore.so
├── common-metal/     # CMake project: deepx_core_metal → libdeepx_core_metal.a
└── old-cppcommon/    # 非正式库，仅原始头文件/源文件集合
```

---

### 1.1 dxlang (`deepxcore`)

| 维度 | 详情 |
|------|------|
| **定位** | 语言层基础库：统一类型系统与工具类 |
| **产出** | `libdeepxcore.so` (SHARED) |
| **外部依赖** | yaml-cpp |
| **使用者** | exop-cpu（通过 `add_subdirectory(../deepxcore)`） |

**提供内容：**

| 路径 | 功能 |
|------|------|
| `src/deepx/dtype/precision.hpp` | Precision 枚举（位图）：Float64~Float4E2M1, Int64~Int4, Bool, String；含 precision_bits / from_string / to_string |
| `src/deepx/dtype/data_category.hpp` | DataCategory 枚举（位图）：Var, Vector, Tensor, ListTensor；含 to_string / from_string |
| `src/deepx/dtype/typespec.hpp` | TypeSpec 联合体：DataCategory + Precision 组合，含 match() / autodtype() |
| `src/stdutil/num.hpp` | is_integer / is_float / is_positive_integer |
| `src/stdutil/string.hpp` | trim / trimspace / escape_markdown |
| `src/stdutil/error.hpp` | NotImplementError / UnsupportedOperationException / TensorShapeError |
| `src/stdutil/fs.hpp` | 文件系统工具 |
| `src/stdutil/time.hpp` | 时间工具 |
| `src/stdutil/vector.hpp` | 向量工具 |
| `src/stdutil/print.hpp` | 打印工具 |

**设计理念**（摘自 README.md）：
- 语言先于执行：先统一语义，再扩展执行器
- IR-first：与 deepxIR 规范对齐
- 稳定契约：跨进程、跨模块、跨后端保持一致语义
- 执行解耦：不绑定 CUDA/Metal/ROCm 等实现细节
- 最小依赖：保持轻量、可移植、可复用

---

### 1.2 common-metal (`deepx_core_metal`)

| 维度 | 详情 |
|------|------|
| **定位** | Metal 平台公共基础设施（**但部分内容平台无关**） |
| **产出** | `libdeepx_core_metal.a` (STATIC) |
| **外部依赖** | Metal.framework, Foundation.framework (Apple only) |
| **使用者** | heap-metal, exop-metal |

**提供内容：**

| 路径 | 功能 | 平台依赖 |
|------|------|----------|
| `include/deepx/shmem/shm_tensor.h` | POSIX 共享内存张量（shm_tensor_create/open/close/unlink） | **无**（纯 POSIX） |
| `src/shmem/shm_tensor.cpp` | mmap/shm_open 实现 | **无**（纯 POSIX） |
| `include/deepx/registry.h` | 抽象 TensorMeta 注册接口（create_or_get/ref_inc/ref_dec/get_meta） | **无**（纯 C++） |
| `include/deepx/metal_device.hpp` | Apple Metal 设备检测 | Apple only |
| `src/metal_device.cpp` | ObjC Metal API 实现 | Apple only |

---

### 1.3 old-cppcommon

| 维度 | 详情 |
|------|------|
| **定位** | 旧版张量类型、TF 框架、算子接口定义 |
| **产出** | 无（非正式库，仅 `include_directories()` 引用） |
| **外部依赖** | 通过使用者间接依赖 STL |
| **使用者** | exop-metal（`include_directories(../old-cppcommon)`）；exop-cpu 内部有完整副本 |

**提供内容：**

| 分类 | 文件 | 功能 |
|------|------|------|
| **核心类型** | `dtype.hpp` | ⚠️ 空文件，已被 dxlang 替代 |
| | `shape.hpp` | Shape 结构体：dtype, shape, strides, size, bytes()；range/rangeParallel/rangeElementwiseParallel |
| | `tensorbase.hpp` | TensorBase：仅持有 Shape |
| | `tensor.hpp` | Tensor\<T\> 模板：data 指针 + newer/deleter/copyer/saver/loader（完整 RAII） |
| **内存管理** | `mem/mem.hpp` | MemBase：args(map) + tensor 存储(map)；addarg/getarg/addtensor/gettensor |
| **TF 框架** | `tf/tf.hpp` | TF 类（name/tftype/args/returns/metadata）；Param；TFMetadata；OpResp（UDP 通信） |
| | `tf/tffactory.hpp` | TfFactory → TFFamily → TFAuthor 多层注册与类型匹配调度 |
| **算子接口** | `tensorfunc/authors.hpp` | Author 标签：default_, miaobyte, cblas, cublas |
| | `tensorfunc/elementwise.hpp` | Dispatcher：add/sub/mul/div/pow/sqrt/log/exp/sin/cos/tan/max/min/equal/less/greater/invert/switch |
| | `tensorfunc/matmul.hpp` | Dispatcher：matmul |
| | `tensorfunc/init.hpp` | Dispatcher：constant/arange/dropout/uniform/normal |
| | `tensorfunc/reduce.hpp` | Dispatcher：reducemax/reducemin/sum/prod |
| | `tensorfunc/changeshape.hpp` | Dispatcher：reshape/transpose/concat/broadcastTo/indexselect/repeat/repeat_interleave |
| | `tensorfunc/io.hpp` | Dispatcher：print |
| **辅助** | `shape_*.hpp/cpp` | Shape 操作实现 |
| | `vector_combination.hpp/cpp` | 向量组合 |
| | `client/udpserver.hpp/cpp` | UDP 服务器（旧通信方式） |
| | `client/unixsocketserver.hpp/cpp` | Unix Socket 服务器 |
| | `client/worker.hpp` | 工作线程 |

---

## 2. 问题诊断

### 问题 1：common-metal 命名不当 — 平台无关代码被锁在 Metal 库中

`shm_tensor`（POSIX 共享内存）和 `Registry`（抽象注册接口）是纯 POSIX/C++ 实现，**不依赖任何 Metal API**。

但 `common-metal/CMakeLists.txt` 强制链接 Metal.framework：

```cmake
find_library(METAL Metal)
find_library(FOUNDATION Foundation)
target_link_libraries(deepx_core_metal PUBLIC ${METAL} ${FOUNDATION})
```

**影响**：任何非 Apple 平台（Linux/Windows）无法使用 `shm_tensor` 和 `Registry`。exop-cpu（CPU 后端）和 op-cuda（CUDA 后端）无法复用这些已经写好的基础设施。

### 问题 2：old-cppcommon 不是正式库

- 无 CMakeLists.txt，无编译产物
- 使用者通过 `include_directories(../old-cppcommon)` 原始包含
- `dtype.hpp` 为空文件，但 `shape.hpp` 仍然 `#include "dtype.hpp"` 同时 `#include "deepx/dtype/precision.hpp"` —— 依赖混乱
- 算子接口（elementwise.hpp 等）依赖 `tensor.hpp` 和 `mem/mem.hpp`，与 MemBase 强耦合
- 不兼容新的 Redis+shm 架构（新架构使用裸指针 + Redis 元数据，而非 Tensor\<T\> + MemBase）

### 问题 3：dxlang 与 old-cppcommon 存在功能重叠

| 功能 | dxlang | old-cppcommon |
|------|--------|---------------|
| 精度类型 | `Precision` 枚举（位图，完善） | `dtype.hpp`（空文件，废弃） |
| 数据类别 | `DataCategory` 枚举（位图） | 无 |
| 类型组合 | `TypeSpec` 联合体 | 无 |
| 字符串工具 | `stdutil/string.hpp` | 无 |
| 数字工具 | `stdutil/num.hpp` | 无 |
| 错误类型 | `stdutil/error.hpp` | 无 |
| Shape 定义 | 无 | `Shape` 结构体 |
| Tensor 定义 | 无 | `Tensor<T>` 模板 |
| 内存管理 | 无 | `MemBase` |
| TF 框架 | 无 | `TF`, `TfFactory` |

### 问题 4：新旧架构割裂

```
旧架构（UDP + MemBase）：
  old-cppcommon → TF::run(mem) → 通过 MemBase 访问 Tensor<T>
  通信：UDP Socket，自定协议
  使用者：exop-cpu（原名 op-ompsimd）

新架构（Redis + shm）：
  common-metal → shm_tensor → mmap 直接访问内存
  通信：Redis BLPOP + JSON
  使用者：exop-metal, heap-metal
```

两者之间没有桥接层。exop-cpu 如果要参与联合调试，其 main.cpp 必须重写为 Redis+shm 架构。

---

## 3. 整合方案

### 3.1 目标目录结构

```
executor/
├── deepx-core/                 # 【新建】统一的平台无关公共库
│   ├── CMakeLists.txt            # → libdeepx_core.a (STATIC)
│   ├── include/deepx/
│   │   ├── dtype/                # ← 从 dxlang 迁移
│   │   │   ├── precision.hpp
│   │   │   ├── data_category.hpp
│   │   │   └── typespec.hpp
│   │   ├── tensor/               # ← 从 old-cppcommon 提炼
│   │   │   ├── shape.hpp
│   │   │   ├── tensor.hpp
│   │   │   └── tensor_base.hpp
│   │   ├── shmem/                # ← 从 common-metal 拆分
│   │   │   └── shm_tensor.h
│   │   └── registry.h            # ← 从 common-metal 拆分
│   └── src/
│       ├── stdutil/              # ← 从 dxlang 迁移
│       │   ├── error.hpp
│       │   ├── fs.hpp / fs.cpp
│       │   ├── num.hpp / num.cpp
│       │   ├── print.hpp
│       │   ├── string.hpp / string.cpp
│       │   ├── time.hpp
│       │   └── vector.hpp
│       ├── shmem/
│       │   └── shm_tensor.cpp
│       └── tensor/
│           └── shape.cpp
│
├── common-metal/                 # 精简为仅 Metal HAL
│   ├── CMakeLists.txt            # → libdeepx_metal_hal.a
│   ├── include/deepx/metal/
│   │   └── metal_device.hpp
│   └── src/metal/
│       └── metal_device.cpp
│
├── dxlang/                       # ⚠️ 代码已迁移，目录保留兼容
│   └── README.md                 # 说明迁移到 deepx-core
│
├── old-cppcommon/                # ⚠️ 逐步拆分
│   ├── tensorfunc/               # 保留算子接口（多后端共享）
│   │   ├── authors.hpp
│   │   ├── elementwise.hpp
│   │   ├── matmul.hpp
│   │   ├── init.hpp
│   │   ├── reduce.hpp
│   │   ├── changeshape.hpp
│   │   └── io.hpp
│   ├── tf/                       # TF 框架（exop-cpu 旧架构依赖）
│   └── client/                   # 旧通信方式（逐步淘汰）
│       └── udpserver / unixsocketserver
```

### 3.2 各执行器依赖关系（整合后）

```
                    ┌─────────────────────────────┐
                    │     deepx-core (STATIC)    │
                    │  dtype / tensor / shmem /    │
                    │  registry / stdutil          │
                    └──────┬──────────┬───────────┘
                           │          │
              ┌────────────┼──────────┼──────────────┐
              │            │          │              │
              ▼            ▼          ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ exop-cpu │ │ op-cuda  │ │ exop-metal │ │heap-metal│
        │  (CPU)   │ │ (CUDA)  │ │ (Metal)  │ │ (Metal)  │
        └──────────┘ └──────────┘ └────┬─────┘ └────┬─────┘
                                       │            │
                                       ▼            ▼
                                ┌────────────────────────┐
                                │ common-metal (HAL only)│
                                │  metal_device.hpp      │
                                └────────────────────────┘
```

---

## 4. 分步执行计划

### Phase 1：创建 deepx-core（本次执行）

**Step 1.1** — 创建目录结构：
```bash
mkdir -p executor/deepx-core/{include/deepx/{dtype,tensor,shmem},src/{stdutil,shmem,tensor}}
```

**Step 1.2** — 从 dxlang 迁移类型系统（直接复制，无需修改）：
```
dxlang/src/deepx/dtype/precision.hpp      → deepx-core/include/deepx/dtype/precision.hpp
dxlang/src/deepx/dtype/data_category.hpp  → deepx-core/include/deepx/dtype/data_category.hpp
dxlang/src/deepx/dtype/typespec.hpp       → deepx-core/include/deepx/dtype/typespec.hpp
dxlang/src/stdutil/*                      → deepx-core/src/stdutil/*
```

**Step 1.3** — 从 common-metal 拆分平台无关部分：
```
common-metal/include/deepx/shmem/shm_tensor.h  → deepx-core/include/deepx/shmem/shm_tensor.h
common-metal/src/shmem/shm_tensor.cpp          → deepx-core/src/shmem/shm_tensor.cpp
common-metal/include/deepx/registry.h          → deepx-core/include/deepx/registry.h
```

**Step 1.4** — 从 old-cppcommon 提炼核心张量类型（需适配）：
- `shape.hpp` → `deepx-core/include/deepx/tensor/shape.hpp`
  - 保留 Shape 结构体核心方法
  - 将 OpenMP 相关方法（rangeParallel 系列）标记为可选/条件编译
- `tensor.hpp` → `deepx-core/include/deepx/tensor/tensor.hpp`
  - 保留 Tensor\<T\> 模板
- `tensorbase.hpp` → `deepx-core/include/deepx/tensor/tensor_base.hpp`
- ⚠️ `dtype.hpp` → **删除**，所有引用改为 `deepx/dtype/precision.hpp`

**Step 1.5** — 编写 `deepx-core/CMakeLists.txt`：
```cmake
cmake_minimum_required(VERSION 3.15)
project(deepx-core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(deepx_core STATIC ${SOURCES})
target_include_directories(deepx_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
find_package(yaml-cpp REQUIRED)
target_link_libraries(deepx_core PUBLIC yaml-cpp)
```

**Step 1.6** — 更新 common-metal/CMakeLists.txt：
```cmake
# 依赖 deepx-core（不再内置 shm_tensor/registry）
if(NOT TARGET deepx_core)
    add_subdirectory(../deepx-core deepx-core)
endif()
# 输出精简为 Metal HAL
add_library(deepx_metal_hal STATIC src/metal/metal_device.cpp)
target_link_libraries(deepx_metal_hal PUBLIC deepx_core ${METAL} ${FOUNDATION})
```

**Step 1.7** — 更新各 executor 的 CMakeLists.txt：
- `exop-cpu`: `add_subdirectory(../deepx-core deepx-core)` 替代 `add_subdirectory(../deepxcore deepxcore)`
- `exop-metal`: 同时依赖 `deepx-core` + `deepx_metal_hal`
- `heap-metal`: 同时依赖 `deepx-core` + `deepx_metal_hal`

### Phase 2：统一类型系统（后续）

- 统一 Precision 字符串表示：全部使用 `"f32"/"f64"/"i32"/"i64"` 等短名称
- 在 `precision.hpp` 中确认 `precision_from_string()` 双向映射完整
- 定义 `deepx::Tensor` 为平台无关的最小接口

### Phase 3：exop-cpu main.cpp 现代化（本次需完成）

exop-cpu 当前的 main.cpp 基于旧架构（UDP + MemBase + TF 框架）。需要新增 Redis+shm 版本：

- 复用 `exop-metal/src/client/main.cpp` 的架构（Redis BLPOP、shm、heartbeat、instance registration）
- 将 Metal GPU kernel 调用替换为 CPU 实现（elementwise binary/unary/scalar、init、comparison、reduce、changeshape）
- `PROG_NAME = "op-cpu"`，与 VM 代码中 `[]string{"exop-metal", "op-cuda", "op-cpu"}` 对齐
- 支持相同的 opcode 集合

### Phase 4：heap-cpu 创建（本次需完成）

基于 `heap-metal/` 创建 `heap-cpu/`：
- 复用 lifecycle + registry 逻辑
- `PROG_NAME = "heap-cpu"`，注册到 `/sys/heap-plat/heap-cpu-{hostname}-{pid}`
- 使用 CPU 内存分配（malloc/mmap）替代 GPU 显存
- 去除 Metal device 检测

---

## 5. 文件变更清单

### 新建
```
executor/deepx-core/CMakeLists.txt
executor/deepx-core/include/deepx/dtype/precision.hpp       ← dxlang
executor/deepx-core/include/deepx/dtype/data_category.hpp   ← dxlang
executor/deepx-core/include/deepx/dtype/typespec.hpp        ← dxlang
executor/deepx-core/include/deepx/shmem/shm_tensor.h        ← common-metal
executor/deepx-core/include/deepx/registry.h                ← common-metal
executor/deepx-core/include/deepx/tensor/shape.hpp          ← old-cppcommon (提炼)
executor/deepx-core/include/deepx/tensor/tensor.hpp         ← old-cppcommon (提炼)
executor/deepx-core/include/deepx/tensor/tensor_base.hpp    ← old-cppcommon (提炼)
executor/deepx-core/src/stdutil/*                           ← dxlang (全部)
executor/deepx-core/src/shmem/shm_tensor.cpp                ← common-metal
executor/deepx-core/src/tensor/shape.cpp                    ← old-cppcommon
executor/heap-cpu/ (整个目录)                                 ← 基于 heap-metal 新建
```

### 修改
```
executor/common-metal/CMakeLists.txt       # 精简：移除 shm_tensor/registry，依赖 deepx-core
executor/exop-cpu/CMakeLists.txt           # 依赖 deepx-core 替代 deepxcore
executor/exop-cpu/src/client/main.cpp      # 新增 Redis+shm 版本
executor/exop-metal/CMakeLists.txt           # 依赖 deepx-core + 精简后的 common-metal
executor/heap-metal/CMakeLists.txt         # 依赖 deepx-core + 精简后的 common-metal
executor/Makefile                          # 添加 build-exop-cpu / build-heap-cpu 目标
Makefile (根)                              # 添加 build-exop-cpu / build-heap-cpu 目标
```

### 废弃/标记
```
executor/dxlang/README.md           # 添加"已迁移至 deepx-core"说明
executor/old-cppcommon/README.md    # 添加拆分说明
```

---

## 6. 迁移兼容性矩阵

| 库 | 当前依赖 | 迁移后依赖 | 影响范围 |
|----|---------|-----------|----------|
| exop-cpu | dxlang (`deepxcore`) | deepx-core (`deepx_core`) | include 路径变更，CMake target 名变更 |
| exop-metal | common-metal + old-cppcommon + dxlang | deepx-core + common-metal (HAL only) | CMake 重构 |
| heap-metal | common-metal | deepx-core + common-metal (HAL only) | CMake 重构 |
| op-cuda | (规划中) | deepx-core | 开箱即用 |
| heap-cpu | (规划中) | deepx-core | 开箱即用 |

---

## 7. 风险与注意事项

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| dxlang 当前编译为 `.so`，deepx-core 建议 `.a` | 低 | exop-cpu 使用 `add_subdirectory` 直接编译，无 ABI 问题 |
| Shape 类依赖 OpenMP（`rangeParallel` 等方法） | 中 | 提炼时使用条件编译 `#ifdef _OPENMP`，或在 deepx-core 中仅保留串行方法 |
| old-cppcommon 与 exop-cpu/src/deepx 有大量重复代码 | 中 | Phase 1 暂不处理，记录为后续去重任务 |
| include 路径变更需全局搜索替换 | 中 | 使用 `grep -rn` 全局扫描，批量 `sed` 替换 |
| TF 框架与新架构不兼容 | 高 | 保留旧 main.cpp 为 `main_legacy.cpp`，新写 Redis+shm 版本 |

### Phase 5：合并完整性审计（✅ 全部完成 2026-04-30）

**Step 5.1** — 审计 dxlang → deepx-core：✅ 已完成
- 逐文件 diff 验证：所有 dtype + stdutil 文件内容一致
- 无遗漏的类型定义或工具函数
- dxlang 源码已删除，仅保留 README.md 说明迁移
- exop-metal 中对 dxlang 的死引用已清理

**Step 5.2** — 审计 old-cppcommon → deepx-core：✅ 已完成（三个阶段）
- 阶段一（核心类型）：删除 5 个文件（dtype.hpp/shape.hpp/tensor.hpp/tensorbase.hpp/shape.cpp）— 已迁移至 deepx-core
- 阶段二（shape 辅助工具）：迁移 11 个文件（shape_changeshape/matmul/reduce/tensorinit/range/vector_combination）
- 阶段三（TF 框架 + 算子接口）：迁移 13 个文件（tensorfunc/8 + tf/4 + mem/1）
- old-cppcommon 整体 `rm -rf` 删除，34 个文件全部迁移完成
- 所有保留代码的 include 已更新为 deepx-core 规范路径
- exop-metal CMakeLists.txt 移除 `include_directories(../old-cppcommon)`

**Step 5.3** — 审计 common-metal → deepx-core：✅ 已完成
- 确认 shm_tensor 和 registry 平台无关代码已全部拆分
- 逐文件 diff 验证：shm_tensor.h 完全一致；registry.h 完全一致；shm_tensor.cpp 仅 include 路径差异（旧 `"deepx/shmem/shm_tensor.h"` → 新 `"deepx/shm_tensor.h"`）
- 删除 common-metal 中已迁移的 3 个文件：registry.h, shmem/shm_tensor.h, shmem/shm_tensor.cpp
- common-metal 精简后仅保留 2 个源文件：metal_device.hpp + metal_device.cpp
- CMakeLists.txt 确认仅编译 metal_device.cpp，构建 `deepx_metal_hal`
- 清理空目录：include/deepx/shmem/, src/shmem/
- 修复 io-metal 中对旧 `"deepx/shmem/shm_tensor.h"` 路径的引用 → `"deepx/shm_tensor.h"`

---

*文档版本: v1.1 | 日期: 2026-04-30 | 作者: claude*
