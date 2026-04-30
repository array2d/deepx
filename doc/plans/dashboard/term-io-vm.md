# VM 对接 dashboard 终端 I/O 方案

## 概述

Dashboard 的 Web 终端通过 WebSocket 连接浏览器。VM 也通过 3 个独立 WebSocket 端点连接 dashboard 实现 I/O。

```
浏览器 xterm.js ←WebSocket→ Dashboard ←WebSocket→ VM (stdout)
浏览器 xterm.js ←WebSocket→ Dashboard ←WebSocket→ VM (stderr)
浏览器 xterm.js ←WebSocket→ Dashboard →WebSocket→ VM (stdin)
```

## 注册位置

Dashboard 启动时将终端注册到 `/sys/term/${name}/` 下：

```
/sys/term/dashboard/stdout → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/stdout"}
/sys/term/dashboard/stderr → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/stderr"}
/sys/term/dashboard/stdin  → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/stdin"}
```

每个 key 是 Redis HASH：
- `type`: `"websocket"` 或 `"file"`
- `detail`: WebSocket URL 或文件路径

## VM 关联终端

VM 通过 vthread 的 `/vthread/<vtid>/term` key 关联终端：

```
SET /vthread/<vtid>/term "dashboard"
```

- 该 key 默认为空字符串，表示无终端（print/cerr/input 静默无操作）
- 值 `"dashboard"` 对应 `/sys/term/dashboard/` 下的 3 个 HASH

## VM 对接步骤

### 1. 终端解析

```
GET /vthread/<vtid>/term           → "dashboard"（空则无终端）
HMGET /sys/term/dashboard/stdout type detail  → "websocket", "ws://..."
HMGET /sys/term/dashboard/stderr type detail  → "websocket", "ws://..."
HMGET /sys/term/dashboard/stdin  type detail  → "websocket", "ws://..."
```

### 2. 连接（按 type 分派）

**websocket 模式**：
```
stdout_conn = WebSocket.connect("ws://127.0.0.1:8080/api/term/stdout")
stderr_conn = WebSocket.connect("ws://127.0.0.1:8080/api/term/stderr")
stdin_conn  = WebSocket.connect("ws://127.0.0.1:8080/api/term/stdin")
```

**file 模式**：
```
stdout → APPEND /path/to/stdout.log
stderr → APPEND /path/to/stderr.log
stdin  → READ   /path/to/stdin.txt
```

### 3. 协议

**纯二进制流，无 JSON 序列化。**

- stdout / stderr：VM → 直接发送原始字节
- stdin：浏览器输入 → VM 读取原始字节

stderr 的数据在到达浏览器之前，dashboard 会自动包裹 ANSI 红色转义序列（`\x1b[31m ... \x1b[0m`），VM 端无需任何处理。

### 4. 数据流

```
创建 vthread 时:
  SET /vthread/42/term "dashboard"

VM print("hello")
  → GET /vthread/42/term → "dashboard"
  → HMGET /sys/term/dashboard/stdout → {type:"websocket", detail:"ws://..."}
  → WebSocket("/api/term/stdout").send("hello\n")
    → Dashboard 收到原始字节 → broadcast 到所有浏览器终端

VM cerr("error")
  → GET /vthread/42/term → "dashboard"
  → HMGET /sys/term/dashboard/stderr → {type:"websocket", detail:"ws://..."}
  → WebSocket("/api/term/stderr").send("error\n")
    → Dashboard 包裹 ANSI red → broadcast

VM input("> ")
  → GET /vthread/42/term → "dashboard"
  → HMGET /sys/term/dashboard/stdout → 发送 prompt "> "
  → HMGET /sys/term/dashboard/stdin → 阻塞读取用户输入
```

## 连接管理

- 每个 WebSocket URL 同一时间一个连接（按 URL 缓存复用）
- 新 VM 连接时复用已有连接
- 浏览器终端随时可以连接/断开，不影响 VM 连接
- VM 断线后，浏览器终端继续正常使用 bash
- `/vthread/<vtid>/term` 为空 → 无终端 → IO 指令静默无操作
