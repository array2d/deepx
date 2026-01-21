# mem-cuda 方案草案

本目录用于设计/实现单机多进程的 GPU Tensor 统一存储面（CUDA IPC），并通过 Redis 做 name → IPC handle 的集中注册与控制。

## 目标
- 单机内多进程共享**可命名**Tensor（同名即同一块 GPU 内存）。
- 通过 Redis 维护 name、shape、dtype、device、IPC handle 等元信息。
- 通过 Redis List 接收创建/获取/删除指令，实现统一的控制面。

## 设计概述
### 1) Redis 元数据（KV/Hash）
对每个 Tensor 名称建立一个 Hash：
- `name`: string
- `dtype`: string（如 f32/i8 等）
- `shape`: string/json（如 "[2,3,4]")
- `device`: int（GPU id）
- `bytes`: int
- `ipc_handle`: binary/base64
- `owner_pid`: int
- `refcount`: int
- `ctime/mtime`: int64

### 2) Redis 指令队列（List）
用 list 作为控制通道（生产者/消费者）：
- list key: `tensor:cmd`
- 指令格式建议为 JSON：
  ```json
  {"op":"create|get|delete", "name":"X", "dtype":"f32", "shape":[2,3], "device":0}
  ```
- 处理流程：
  - **create**: 分配 GPU 内存 → `cudaIpcGetMemHandle` → 写入 Hash
  - **get**: 读取 Hash → `cudaIpcOpenMemHandle`
  - **delete**: `refcount--`，为 0 时释放 GPU 内存并删除 Hash

### 3) CUDA IPC 基本流程
- `cudaIpcGetMemHandle`：将 `cudaMalloc` 的指针导出为 handle
- `cudaIpcOpenMemHandle`：其他进程映射同一块 GPU 内存
- 仅限**同机**；需保证 device id 一致
- 跨 stream 写读，需要显式同步（事件/流同步策略）

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

## 目录建议
```
mem-cuda/
  README.md
  doc/
  src/
    registry/        # Redis 元数据与命令处理
    allocator/       # 显存池实现或适配层
    ipc/             # cudaIpcGet/Open 封装
```

## 后续工作清单
- [ ] 选定显存池方案（RMM / CNMeM / CUB / 自研）
- [ ] 定义 Redis 数据结构与命令协议
- [ ] 编写 IPC 封装与单机多进程 demo
- [ ] 建立错误恢复与 GC 机制
