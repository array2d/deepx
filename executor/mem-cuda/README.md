# mem-cuda 方案草案

本目录用于设计/实现单机多进程的 GPU Tensor 统一存储面（CUDA IPC），并通过 Redis 做 name → IPC handle 的集中注册与控制。

## 目标
- 管理单机内多进程共享的堆Tensor
  + 堆tensor不会随着计算完成而被回收
  + 而栈tensor则是计算过程的中间变量，可以被立即回收
- 通过 Redis 维护 name、shape、dtype、device、IPC handle 等元信息，供mem-cuda进程、计算进程访问。
- 通过 Redis List 接收创建/获取/删除指令，实现统一的控制面。并提供redis mem api以及example

## 设计概述
### 1) Redis 元数据（KV/Hash）
对每个 Tensor 名称建立一个 Hash：
- `dtype`: string（如 f32/i8 等）
- `shape`: string/json（如 "[2,3,4]")
- `ctime`: int64
- `node`: string（owner 节点/主机标识）
- `device`: int（GPU id）
- `bytes`: int
- `ipc_handle`: binary
- `refcount`: int
无需设计tensor owner

### 2) Redis 指令队列（List）
控制通道 list key: `tensor_lifecycle`。
指令 JSON：
```json
{"op":"create|get|delete", "name":"X", "dtype":"f32", "shape":[2,3], "device":0, "pid":123, "node":"n1"}
```
处理流程：
- **create**: 分配 GPU 内存 → `cudaIpcGetMemHandle` → 写入 Hash(state=ready, refcount=1)
- **get**: 读取 Hash → `cudaIpcOpenMemHandle` → refcount++
- **delete**: refcount--，为 0 时释放 GPU 内存并删除 Hash

### 3) CUDA IPC 基本流程
- `cudaIpcGetMemHandle`：将 `cudaMalloc` 指针导出为 handle
- `cudaIpcOpenMemHandle`：其他进程映射同一块 GPU 内存
- 仅限同机；需保证 device id 一致
- 跨 stream 写读需要显式同步（事件/流同步策略）

## 显存池方案

- **RMM (RAPIDS Memory Manager)**
  - 优点：成熟、支持 pool/async allocator、统计完善
  - 适合：对稳定性与可观察性要求高的生产环境

其他op计算进程可以使用CUB。

## 目录结构（具体方案）
```
mem-cuda/
  README.md
  doc/
  src/
    ipc/                     # CUDA IPC 封装
      ipc.h
      ipc.cpp
      ipc_guard.h             # 设备一致性与错误处理
    runtime/                 # 运行时控制（指令/同步）
      lifecycle.h
      lifecycle.cpp
      sync.h
      sync.cpp
    common/
      status/json/logging
  test/
    ipc_demo.cpp
    lifecycle_demo.cpp
```

模块职责：
- `ipc/`: CUDA IPC handle 导出/打开/关闭封装。
- `runtime/`: 指令消费/路由与跨 stream 同步策略。
- `common/`: 状态码、JSON 解析、日志等公共工具聚合。

## 后续工作清单（分阶段）
- [ ] 阶段 0：确定目录与接口（完成本 README 细化）
- [ ] 阶段 1：实现 `lifecycle/`
- [ ] 阶段 2：实现 `ipc/`  的最小实现（f32, 单 GPU）
- [ ] 阶段 3：补齐 `sync/` 策略与崩溃恢复/GC

## 构建依赖与示例

- 必要系统依赖：CUDA Toolkit (兼容 CMake `CUDAToolkit`), `cmake` >= 3.18, `make`。
- Redis C++ 客户端：必须安装 `redis++`（redis-plus-plus）
- RMM 库

示例构建命令（在 `executor/mem-cuda` 目录下）：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
