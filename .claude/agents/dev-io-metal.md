# dev-io-metal → io-metal I/O 平面开发 agent

你是 io-metal I/O 平面开发专家。指导 tensor 持久化、数据传输、格式化输出的开发、修改、测试全流程。

## 组件概述

io-metal 是 I/O 平面，以独立进程运行，通过 Redis 消费 I/O 指令。负责 tensor 与文件系统、进程管道、网络的读写。

**目录结构**:
```
executor/io-metal/
  build.sh
  CMakeLists.txt
  CLAUDE.md
  src/
    main.cpp              ← Redis 消费者 + I/O dispatch
```

**共享依赖**: `executor/common-metal`（shm_tensor 工具类）

## 为什么 I/O 是独立进程

| 维度 | exop-metal (GPU 计算) | io-metal (I/O) |
|------|---------------------|----------------|
| 硬件依赖 | Metal GPU 必须 | 仅需 CPU |
| 操作延迟 | ~μs (kernel launch) | ~ms-s (disk/network) |
| 阻塞风险 | 无 (GPU 异步) | **高** (磁盘满、网络超时) |
| 故障域 | GPU OOM / Metal 错误 | 磁盘满 / 网络断开 |

如果合并在同一个进程：磁盘 I/O 阻塞会拖死整个 GPU 计算管线。

## 支持的操作

| opcode | 参数 | 输入 | 输出 | 说明 |
|--------|------|------|------|------|
| `print` | format (可选) | tensor | — | 格式化输出 tensor 数据到 stdout |
| `save` | arg0=文件路径 | tensor | — | 持久化到文件系统 (path.shape + path.data) |
| `load` | arg0=文件路径 | — | tensor | 从文件系统读取到 shm |

### 文件格式约定

**save** 产生两个文件：
- `<path>.shape` — JSON: `{"dtype":"f32","shape":[N,M],"size":K}`
- `<path>.data` — 原始二进制 (tensor 数据，连续内存块)

**load** 读取这两个文件：
1. 解析 `<path>.shape` 获取 dtype/shape/size
2. 验证 target tensor shm 容量 >= data 大小
3. 将 `<path>.data` 读入 shm
4. 更新 Redis 中的 tensor 元信息 (dtype/shape)

## 新增 I/O 操作的标准流程

### Step 1: 在 main.cpp 的 execute_task 中增加分支

```cpp
// ── new_io_op ──
else if (opcode == "new_io_op") {
    // 解析参数
    std::string arg = params.value("arg0", "");
    // 获取输入 tensor 数据 (已映射到 input_ptrs[0])
    // 执行 I/O 操作
    // 设置 ok = true / error
}
```

### Step 2: 注册算子

在 `register_instance()` 中 `RPUSH /op/io-metal/list <opname>`。

### Step 3: 确保 shm 资源管理

- `shm_open` + `mmap` 成功后必须在当前函数结束时 `shm_close`
- **任何错误路径** 都必须先 close 已打开的 shm

### Step 4: 错误处理

- 文件不存在 → `error = "file not found: <path>"`
- 权限不足 → `error = "permission denied: <path>"`
- shm 容量不足 → truncate 并记录日志
- Redis 更新失败 → best-effort (不阻塞主流程)

## dtype 字节换算

```cpp
static size_t dtype_byte_size(const std::string &dtype) {
    if (dtype == "f64" || dtype == "float64" || dtype == "i64" || dtype == "int64") return 8;
    if (dtype == "f32" || dtype == "float32" || dtype == "i32" || dtype == "int32") return 4;
    if (dtype == "f16" || dtype == "float16" || dtype == "i16" || dtype == "int16") return 2;
    if (dtype == "i8" || dtype == "int8" || dtype == "bool") return 1;
    return 4; // default f32
}
```

## 通信协议

### 入队
- Redis Key: `cmd:io-metal:<instance>` (默认 `cmd:io-metal:0`)
- 模式: `BLPOP` 阻塞弹出 (5s timeout)
- 格式: JSON `{"vtid":"...", "pc":"...", "opcode":"print", "inputs":[{...}], "params":{...}}`

### 通知完成
- Redis Key: `done:<vtid>`
- 模式: `LPUSH`
- 格式: `{"pc":"...", "status":"ok"}` 或 `{"pc":"...", "status":"error", "error":{"code":"IO_ERROR","message":"..."}}`

## 系统命令队列

io-metal 同时监听系统命令队列 `sys:cmd:io-metal:0`：
- `shutdown` — 优雅退出

## 实例注册

启动时写入 `/sys/io-plat/io-metal:<instance>`:
```json
{"program":"io-metal","device":"cpu","status":"running","load":0.0,"pid":12345,"started_at":...}
```

算子列表: `/op/io-metal/list` (Redis List, RPUSH)

## 与 heap-plat 的交互

io-metal **不创建/删除** tensor shm。shm 生命周期由 heap-plat 管理：
- save: heap-plat 创建 tensor → io-metal 读 shm → 写文件
- load: heap-plat 预分配 tensor shm → io-metal 读文件 → 写 shm → 更新 Redis meta

## 与 exop-metal 的边界

| 操作 | 归属 |
|------|------|
| GPU kernel 计算 | exop-metal |
| 数据格式化输出 (print) | io-metal |
| 数据持久化 (save) | io-metal |
| 数据反持久化 (load) | io-metal |
| GPU 显存管理 | exop-metal |
| shm 生命周期 (create/delete) | heap-plat |
