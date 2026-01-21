# mem-cuda 方案草案

本目录用于设计/实现单机多进程的 GPU Tensor 统一存储面（CUDA IPC），并通过 Redis 做 name → IPC handle 的集中注册与控制。

## 目标
- 单机内多进程共享**可命名**Tensor（同名即同一块 GPU 内存）。
- 通过 Redis 维护 name、shape、dtype、device、IPC handle 等元信息。
- 通过 Redis List 接收创建/获取/删除指令，实现统一的控制面。

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
你的需求是：**参考 PyTorch 的显存池管理**。这里给出两种落地路线：

### 方案 A：接入成熟开源显存池（推荐）
可选项目：
- **RMM (RAPIDS Memory Manager)**
  - 优点：成熟、支持 pool/async allocator、统计完善
  - 适合：对稳定性与可观察性要求高的生产环境
- **CNMeM (NVIDIA)**
  - 优点：轻量、易集成
  - 适合：需要最小依赖的场景
- **CUB caching allocator**
  - 优点：性能好、实现简单
  - 适合：希望直接嵌入 CUDA 代码路径

> 选择建议：优先 RMM；想保持最小依赖可用 CNMeM 或 CUB。

### 方案 B：自研简化版显存池（AI 方案）
如果不引入外部依赖，可先实现一个简化版池：
- 维护按 size 分桶的 free-list（如 1MB、2MB、4MB…）
- 分配时优先复用空闲块，不足时 `cudaMalloc` 新块
- 回收时挂回 free-list，不立刻 `cudaFree`
- 支持 `recordStream` / `event` 延迟回收，避免跨流释放风险

**建议先实现 MVP**：
1) 单 GPU
2) 只支持 `create/get/delete`
3) dtype 限定 f32
4) 单进程先跑通，再放开多进程 + IPC

## 安全与一致性
- Redis 写入与 refcount 需要原子操作（Lua 脚本/事务）
- 崩溃恢复：定期清理 owner_pid 不存在的条目
- IPC handle 需与 device id 配对，否则会映射失败

## 目录结构（具体方案）
```
mem-cuda/
  README.md
  doc/
    design.md                # 细化设计文档与协议约束
    redis-schema.md          # Redis KV/Hash/List 结构定义
    ipc.md                   # CUDA IPC 约束、时序与同步策略
    allocator.md             # 显存池方案与接口
  src/
    registry/                # Redis 元数据与命令处理
      redis_client.h
      redis_client.cpp
      registry.h
      registry.cpp
      lua_scripts/            # 原子脚本
        create_or_get.lua
        ref_inc.lua
        ref_dec.lua
        gc_sweep.lua
    ipc/                     # CUDA IPC 封装
      ipc.h
      ipc.cpp
      ipc_guard.h             # 设备一致性与错误处理
    allocator/               # 显存池实现或适配层
      allocator.h
      cuda_pool.cpp
      rmm_adapter.cpp         # 可选
      cnmem_adapter.cpp       # 可选
    runtime/                 # 运行时控制（指令/同步）
      lifecycle.h
      lifecycle.cpp
      sync.h
      sync.cpp
    common/
      status/json/logging
  test/
    ipc_demo.cpp
    registry_demo.cpp
    lifecycle_demo.cpp
  tools/
    memcuda_ctl.cpp           # CLI 工具（create/get/delete/list）
```

模块职责：
- `registry/`: Redis 协议、Lua 原子操作、Hash 读写。
- `ipc/`: CUDA IPC handle 导出/打开/关闭封装。
- `allocator/`: 统一分配接口；可切换 RMM/CNMeM/自研。
- `runtime/`: 指令消费/路由与跨 stream 同步策略。
- `common/`: 状态码、JSON 解析、日志等公共工具聚合。

## 后续工作清单（分阶段）
- [ ] 阶段 0：确定目录与接口（完成本 README 细化）
- [ ] 阶段 1：实现 `registry/` + Redis Lua 原子脚本
- [ ] 阶段 2：实现 `ipc/` + `allocator/` 的最小实现（f32, 单 GPU）
- [ ] 阶段 3：实现 `lifecycle/` worker 与 `tools/` CLI
- [ ] 阶段 4：补齐 `sync/` 策略与崩溃恢复/GC

## 构建依赖与示例

- 必要系统依赖：CUDA Toolkit (兼容 CMake `CUDAToolkit`), `cmake` >= 3.18, `make`。
- Redis C 客户端：推荐安装 `hiredis`（用于底层连接）。可选：`redis++`（C++ wrapper）。
- 可选：RMM/CNMeM 库（若启用对应 adapter）。

示例构建命令（在 `executor/mem-cuda` 目录下）：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_HIREDIS=ON -DUSE_REDISPP=OFF -DUSE_RMM=OFF
make -j$(nproc)
```

常用 CMake 选项：
- `-DUSE_HIREDIS=ON|OFF`：是否链接 hiredis（默认 ON）。
- `-DUSE_REDISPP=ON|OFF`：是否启用 redis++（需要事先安装）。
- `-DUSE_RMM=ON|OFF`：启用 RMM 适配（需要额外提供 RMM 的 include/link 设置）。

