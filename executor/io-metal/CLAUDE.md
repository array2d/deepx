# io-metal 开发约束

> io-metal 的职责边界。哪些能做，哪些**绝对不能碰**。

---

## 1. io-metal 是什么

在 DeepX 元程架构中，io-metal 是 **I/O 平面**——负责 tensor 与文件系统、进程管道、网络的读写。

它只做一件事：**被动消费 I/O 指令 → 读写数据 → 通知完成**。

io-metal 是"无状态的 I/O 执行器"——它不关心 tensor 的计算语义，只关心如何把数据持久化或传输。

---

## 2. 为什么 I/O 要与 GPU 计算分离

| 维度 | op-metal (GPU 计算) | io-metal (I/O) |
|------|---------------------|----------------|
| 硬件依赖 | Metal GPU 必须 | 仅需 CPU |
| 操作延迟 | ~μs (kernel launch) | ~ms-s (disk/network) |
| 阻塞风险 | 无 (GPU 异步) | **高** (磁盘满、网络超时) |
| 失败影响 | GPU OOM / Metal 错误 | 磁盘满 / 网络断开 |

**如果合并在同一个进程**：磁盘 I/O 阻塞会拖死整个 GPU 计算管线。

---

## 3. 允许做的事（白名单）

| 操作 | 允许 | 说明 |
|------|------|------|
| BLPOP `cmd:io-metal:*` | ✅ | 消费 VM 发来的 I/O 指令 |
| GET Redis 获取 tensor 元信息 | ✅ | 仅限 inputs/outputs 的 dtype/shape/shm_name/byte_size |
| shm_open + mmap 映射 tensor 内存 | ✅ | 根据 shm_name 获取 CPU 指针，读/写 tensor 数据 |
| 写文件 (save) | ✅ | 将 tensor shape + data 持久化到文件系统 |
| 读文件 (load) | ✅ | 从文件系统读取 tensor shape + data 到 shm |
| 输出到 stdout (print) | ✅ | 格式化打印 tensor 数据 |
| LPUSH `done:<vtid>` 通知完成 | ✅ | 格式: {pc, status:"ok"\|"error", error?} |
| SET `/sys/io-plat/io-metal:0` | ✅ | 启动时注册进程状态 |
| DEL `/sys/io-plat/io-metal:0` | ✅ | 退出时注销 |

---

## 4. 禁止做的事（黑名单）

### 4.1 绝对禁止：GPU 计算

| 操作 | 禁止 | 原因 |
|------|------|------|
| Metal GPU kernel 调用 | ❌ | I/O 平面不需要 GPU |
| MTLBuffer / MTLDevice 操作 | ❌ | 这是 op-metal 的职责 |
| 修改 tensor 数据（计算） | ❌ | io-metal 只搬运数据，不改变数据 |

### 4.2 禁止：越权修改其他组件

| 操作 | 禁止 | 原因 |
|------|------|------|
| 修改 VM 的 vthread 状态 | ❌ | `/vthread/*` 是 VM 的私有空间 |
| 修改 heap-plat 的分配记录 | ❌ | heap-plat 管理 `/heap/*` |
| 消费 `done:*` 队列 | ❌ | done 是 VM 消费的 |
| 生产 `cmd:op-metal:*` / `cmd:heap-metal:*` | ❌ | 只消费 `cmd:io-metal:*` |
| 创建/删除 tensor shm | ❌ | 这是 heap-plat 的职责 |

### 4.3 禁止：修改 tensor 语义

| 操作 | 禁止 | 原因 |
|------|------|------|
| 修改 dtype | ❌ | 只原样读写 |
| 修改 shape | ❌ | 只原样读写 |
| 类型转换 / cast | ❌ | 这是 op-plat 的职责 |

---

## 5. io-metal 的通信边界

```
VM                        io-metal               heap-metal
 │                            │                       │
 │── PUSH cmd:io-metal:0 ──→  │                       │
 │                            │── GET /data/x ────→ Redis
 │                            │←── {shm_name,...} ── Redis
 │                            │── shm_open("/deepx_t_xxx") → read data
 │                            │── write to file / stdout
 │                            │── LPUSH done:1 ────→ Redis
 │←── BLPOP done:1 ───────── Redis
 │── PC++ 继续                │
```

**io-metal 的边界：**
- 入: `cmd:io-metal:*` 队列 + Redis GET（tensor 元信息）
- 出: `done:<vtid>` 队列
- 内部: shm 映射 → 读/写数据 → 返回

---

## 6. 支持的操作

| opcode | 参数 | 输入 | 输出 | 说明 |
|--------|------|------|------|------|
| `print` | format (可选) | tensor | — | 格式化输出到 stdout |
| `save` | arg0=文件路径 | tensor | — | 持久化到文件系统 (path.shape + path.data) |
| `load` | arg0=文件路径 | — | tensor | 从文件系统读取到 shm |

### 文件格式

**save** 产生两个文件：
- `<path>.shape` — JSON: `{"dtype":"f32","shape":[N,M],"size":K}`
- `<path>.data` — 原始二进制 (tensor 数据)

**load** 读取这两个文件，将数据写入目标 tensor 的 shm 区域。

---

## 7. 允许的日志输出

| 场景 | 允许 |
|------|------|
| 启动: Redis 地址、监听队列 | ✅ 一次性 `std::cout` |
| 启动: CWD、进程 PID | ✅ 一次性 |
| 致命错误: Redis 连接失败 | ✅ `std::cerr` + 重试 |
| 退出: shutdown complete | ✅ 一次性 |
| save/load: 文件路径 + dtype + 元素数 | ✅ 一次性（低频操作） |
| print: 格式化 tensor 数据 | ✅ 这是 print 的职责本身 |
| 每次指令的低级诊断 | ❌ 高频冗余 |
| shm 操作成功日志 | ❌ 成功不输出，失败走 error 通知 |

---

## 8. 与 deepxctl 的关系

| 谁做什么 | deepxctl | io-metal |
|---------|----------|----------|
| 启动 io-metal 进程 | ✅ | — |
| 注册 I/O 算子 | — | ✅ |
| 连接 Redis | — | ✅ |
| 消费 I/O 指令 | — | ✅ |
| 执行 I/O 操作 | — | ✅ |
| 验证输出结果 | ✅ (通过轮询 vthread status) | ❌ |
| 清理子进程 | ✅ | — |
