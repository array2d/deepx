# deepxctl 开发约束

> deepxctl 的职责边界。哪些逻辑 deepxctl 能做，哪些**绝对不能碰**。

---

## 1. deepxctl 是什么

deepxctl 是 DeepX 元程系统的**进程编排工具**。它不实现任何计算/存储/调度逻辑，
只负责**子进程生命周期管理**和**Redis 状态的检查性读写**。

deepxctl = 启动脚本的 Go 化替代品。

---

## 2. 允许做的事（白名单）

### 2.1 子进程管理

| 操作 | 允许 | 说明 |
|------|------|------|
| `exec.Command` 启动 op-plat | ✅ | 调用已编译的二进制 |
| `exec.Command` 启动 heap-plat | ✅ | 同上 |
| `exec.Command` 启动 VM | ✅ | 同上 |
| `exec.Command` 调用 loader | ✅ | 加载 dx 源码到 `/src/func/` |
| 给子进程传递参数（redis addr 等） | ✅ | |
| 捕获子进程 stdout/stderr | ✅ | 仅用于日志展示 |
| SIGTERM / SIGKILL 子进程 | ✅ | 进程清理 |
| 检测子进程退出状态 | ✅ | |

### 2.2 Redis 操作（仅限读取状态 + vthread 管理）

| 操作 | 允许 | 说明 |
|------|------|------|
| `PING` | ✅ | 检测 Redis 可达 |
| `FLUSHDB` | ✅ | 重置测试环境（仅限开发端口 16379） |
| `DBSIZE` | ✅ | 验证重置结果 |
| `GET /sys/op-plat/*` | ✅ | 检查 op-plat 就绪状态 |
| `GET /sys/heap-plat/*` | ✅ | 检查 heap-plat 就绪状态 |
| `GET /sys/vm/*` | ✅ | 检查 VM 就绪状态 |
| `GET /vthread/<vtid>` | ✅ | 轮询 vthread 执行状态 |
| `SET /vthread/<vtid>` | ✅ | 创建 vthread + 设置 init 状态 |
| `SET /vthread/<vtid>/[0,0]` | ✅ | 设置 vthread 入口 CALL 指令 |
| `INCR /sys/vtid_counter` | ✅ | 分配 vthread ID |
| `LPUSH notify:vm` | ✅ | 唤醒 VM 拾取新 vthread |
| `GET /src/func/<name>` | ✅ | 验证 dx 加载是否成功 |
| `KEYS /src/func/*` | ✅ | 列出已注册函数（status 命令用） |

### 2.3 构建

| 操作 | 允许 | 说明 |
|------|------|------|
| 调用 `executor/*/build.sh` | ✅ | 直接 exec 现有脚本 |
| 检测二进制是否存在 | ✅ | 跳过已构建的组件 |

### 2.4 平台检测

| 操作 | 允许 | 说明 |
|------|------|------|
| 检测 GOOS | ✅ | 判断 macOS vs Linux |
| 检测 Metal 可用 | ✅ | Objective-C 小工具或 go 绑定 |
| 检测 nvidia-smi | ✅ | 判断 CUDA 可用 |

---

## 3. 禁止做的事（黑名单）

### 3.1 绝对禁止：实现组件内部逻辑

> deepxctl 不是 VM，不是 op-plat，不是 heap-plat。

| 操作 | 禁止 | 原因 |
|------|------|------|
| 实现任何 tensor 计算 | ❌ | 这是 op-plat 的职责 |
| 分配/释放 shm | ❌ | 这是 heap-plat 的职责 |
| 翻译 dxlang 指令 | ❌ | 这是 VM 的职责 |
| 执行 CALL/RETURN | ❌ | 这是 VM 的职责 |
| 路由算子到 op-plat | ❌ | 这是 VM route 包的职责 |
| 解析 dxlang 语法 | ❌ | 这是 loader + VM ir 包的职责 |
| 注册算子列表 | ❌ | 这是 op-plat 自行注册的 |
| 生产 Redis 命令队列消息 | ❌ | 这是 VM 的职责（push 到 cmd:*） |
| 消费 Redis 命令队列 | ❌ | 这是 op-plat/heap-plat 的职责 |

### 3.2 禁止：修改其他组件

| 操作 | 禁止 | 原因 |
|------|------|------|
| 修改 op-plat 源码 | ❌ | 跨组件边界 |
| 修改 heap-plat 源码 | ❌ | 跨组件边界 |
| 修改 VM 源码 | ❌ | 跨组件边界 |
| 修改 build.sh 脚本 | ❌ | 它们是独立的构建原语 |
| 修改 CMakeLists.txt | ❌ | 构建系统属于各组件 |

### 3.3 禁止：引入新的通信协议

| 操作 | 禁止 | 原因 |
|------|------|------|
| deepxctl 与 VM 直接通信 | ❌ | 所有通信通过 Redis KV 空间 |
| deepxctl 与 op-plat 直接通信 | ❌ | 同上 |
| 自定义 socket/gRPC/HTTP API | ❌ | 不引入额外协议 |
| 新增 Redis key 模式 | ❌ | 只能使用已定义的 key 模式 |

### 3.4 禁止：在生产环境自动 FLUSHDB

| 操作 | 禁止 | 原因 |
|------|------|------|
| 在非 16379 端口自动 FLUSHDB | ❌ | 可能误删生产数据 |
| 无确认直接清除非本地 Redis | ❌ | 同上 |

---

## 4. deepxctl 的 Redis key 使用边界

deepxctl 只能**读**以下 key（状态检查）：

```
读取:
  /sys/op-plat/metal:0        → 检查 op-plat 是否就绪
  /sys/heap-plat/metal:0      → 检查 heap-plat 是否就绪
  /sys/vm/0                   → 检查 VM 是否就绪
  /src/func/<name>            → 验证 dx 加载成功
  /vthread/<vtid>             → 轮询执行状态 (pc, status)
  /op/<program>/list          → (可选) 验证算子注册
```

deepxctl 只能**写**以下 key（vthread 生命周期）：

```
写入:
  /sys/vtid_counter           → INCR 分配 vthread ID
  /vthread/<vtid>             → SET 创建 vthread (pc + status)
  /vthread/<vtid>/[0,0]       → SET 入口指令
  notify:vm                   → LPUSH 唤醒 VM
```

deepxctl **绝对不能**读写的 key：

```
禁止:
  /vthread/<vtid>/[*,*]       → 指令坐标 (这是 VM 的私有格式)
  /op/*/func/*                → 编译层 (这是 VM + 编译器的)
  cmd:op-*                    → 命令队列 (这是 VM 生产、op-plat 消费)
  cmd:heap-*                  → 命令队列 (这是 VM 生产、heap-plat 消费)
  done:*                      → 完成通知 (这是 VM 消费)
  /lock/*                     → 锁 (这是 VM 管理)
  堆变量 (任意非保留路径)       → tensor 元信息 (这是 pysdk + heap-plat 管理)
```

---

## 5. 命令架构

deepxctl 将生命周期拆分为三个独立命令：

```
deepxctl boot       → 构建 + 启动 op-metal、heap-metal、VM，写入 PID 文件
deepxctl run a.dx   → 检测 boot 状态 → 加载 dx → 创建 vthread → 轮询等待结果
deepxctl shutdown   → 有序退出: plats → VM → 心跳验证 → 清理 PID 文件
```

### `deepxctl run` 执行流程

```
deepxctl run xxx.dx [--rm]
│
│  (deepxctl 负责的部分)
│
├─ [1/3] Check services ─────── 检查 boot PID 文件 + Redis 服务就绪
├─ [2/3] Load dx ────────────── exec loader 二进制 (子进程加载 .dx 到 /src/func/)
├─ [3/3] Execute ──────────────
│   ├─ create vthread ───────── SET /vthread/<vtid> (初始状态)
│   ├─ wake VM ──────────────── LPUSH notify:vm
│   │                            │
│   │                            ▼ (VM 接手 — deepxctl 不参与)
│   │                       VM 拾取 → CALL 翻译 → dispatch → op/heap → PC++
│   │
│   └─ poll status ──────────── GET /vthread/<vtid> (轮询 status)
│       ├─ done  → print result ✓
│       └─ error → print error   ✗
│
└─ [--rm] Cleanup (可选)
    ├─ FLUSHDB ───────────────── 重置 Redis KV 空间
    └─ ExecShutdown ──────────── 复用 shutdown 逻辑: plats → VM → 清理
```

**关键分界线**：deepxctl 在 `notify:vm` 之后就不再参与执行——后续所有步骤（CALL 翻译、
指令 dispatch、op 执行、done 通知）都是 VM/op-plat/heap-plat 之间通过 Redis 的协作。
deepxctl 只是**旁观**：轮询 `/vthread/<vtid>` 的 status 字段，直到 `done` 或 `error`。

### `--rm` 一键清理

`deepxctl run a.dx --rm` 在 dx 代码执行成功后自动：
1. **FLUSHDB** — 重置 Redis KV 空间
2. **shutdown** — 复用 `deepxctl shutdown` 的完整退出逻辑（Redis sys:shutdown 命令 → 心跳验证 → 清理 PID 文件 → OS 信号兜底）

等价于手动执行：
```bash
deepxctl run a.dx && make reset-redis && deepxctl shutdown
```

---

## 6. 不允许的"快捷方式"

以下是一些看似方便但**绝不能做**的事情：

| 快捷方式 | 为什么不行 |
|---------|-----------|
| 直接在 deepxctl 里解析 dxlang 找入口函数 | loader + VM 已有解析逻辑，deepxctl 不应重复实现语法解析 |
| 直接 SET `/vthread/<vtid>/[0,1]` 等详细指令 | VM 的 CALL eager 翻译负责展开指令坐标，deepxctl 只负责最顶层的一个 CALL |
| 直接 LPUSH `cmd:op-metal:0` | 命令队列由 VM 生产，deepxctl 绕过 VM 会破坏调度逻辑 |
| 通过 `done:<vtid>` 轮询完成 | VM 消费 done 队列后更新 `/vthread/<vtid>` status，deepxctl 应该读 status 而非直接消费 done 队列 |
| 读取 `/vthread/<vtid>/[*,*]` 指令 | 这是 VM 的内部数据格式，外部不应依赖 |
| 给 VM 发自定义信号 | 所有通信必须走 Redis KV 空间 |

---

## 7. 入口函数约定

deepxctl 创建 vthread 时，写入的指令是**一个顶层 CALL**：

```
/vthread/<vtid>           = {"pc":"[0,0]","status":"init"}
/vthread/<vtid>/[0,0]     = "<entry_func_name>"    ← CALL 指令的操作码
/vthread/<vtid>/[0,1]     = "./ret"                ← 返回值槽位
```

VM 执行到 `[0,0]` 时：
1. 识别 `<entry_func_name>` 不是内置关键字
2. 检查 `/src/func/<entry_func_name>` 存在
3. 触发 CALL eager 翻译，展开函数体到子栈 `[0,0]/[0,0]`, `[0,0]/[1,0]`...
4. 继续逐条执行

**入口函数名确定规则**：
1. 文件中有 `def main` → 用 `main`
2. 只有一个 `def` → 用那个名字
3. 多个 `def` 且无 `main` → 报错，要求 `--entry` 指定

> 入口函数名从文件名推断（loader 已实现命名逻辑），deepxctl 调用 loader 后
> 可以 GET `/src/func/*` 的 KEYS 结果确定有哪些函数可用。

---

## 8. 实现语言和依赖

| 项 | 选择 | 约束 |
|----|------|------|
| 语言 | Go | 与 VM 一致，单二进制 |
| Redis 客户端 | `go-redis/v9` | 与 VM 相同依赖 |
| 进程管理 | `os/exec` | Go 标准库 |
| CLI 框架 | 能跑就行（flag 包即可） | MVP 不引入重型框架 |
| 配置文件 | 硬编码先（后续 YAML） | MVP 阶段不引入 viper |

---

## 9. 当前文件清单

```
tool/deepxctl/
├── main.go
├── go.mod
├── cmd/
│   ├── boot.go              ← boot 子命令 (构建 + 启动服务)
│   ├── run.go               ← run 子命令 (加载 dx + 创建 vthread + 轮询)
│   ├── shutdown.go          ← shutdown 子命令 (有序退出服务)
│   └── common.go            ← 共享打印/辅助函数
├── internal/
│   ├── redis/redis.go       ← 连接 + FLUSHDB + 状态检查 + vthread 管理
│   ├── builder/builder.go   ← exec build.sh
│   ├── process/manager.go   ← 子进程生命周期
│   └── executor/executor.go ← vthread 创建 + 轮询
└── tensor/                  ← tensor 文件操作 (print/save/load)
```

不做的：
- YAML 配置文件解析
- JSON 输出格式（结构化）
- 多平台自动检测（先只支持 metal）
- 守护进程模式
- 远程 Redis TLS
