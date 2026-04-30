# VM 对接 dashboard 终端 I/O 方案

## 概述

Dashboard 的 Web 终端通过 WebSocket 连接浏览器。VM 也通过 WebSocket 连接 dashboard 实现 I/O。

```
浏览器 xterm.js ←WebSocket→ Dashboard ←WebSocket→ VM
```

## 注册位置

Dashboard 启动时写入：

```
/sys/term/dashboard/stdout → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/io"}
/sys/term/dashboard/stderr → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/io"}
/sys/term/dashboard/stdin  → HASH {type: "websocket", detail: "ws://127.0.0.1:8080/api/term/io"}
```

每个 key 是 Redis HASH：
- `type`: `"websocket"`
- `detail`: WebSocket URL，三个 stream 共用一个端点

## VM 对接步骤

### 1. 发现终端

```
KEYS /sys/term/*/stdout
HGET /sys/term/dashboard/stdout type  → "websocket"
HGET /sys/term/dashboard/stdout detail → "ws://127.0.0.1:8080/api/term/io"
```

### 2. 连接 WebSocket

```
WebSocket.connect("ws://127.0.0.1:8080/api/term/io")
```

### 3. 发送 JSON 消息

```json
{"stream":"stdout","data":"hello world\n"}
{"stream":"stderr","data":"error message\n"}
```

- `stream`: `"stdout"` 或 `"stderr"`
- `data`: 输出内容

### 4. 数据流

```
VM print "hello"
  → WebSocket send {"stream":"stdout","data":"hello\n"}
    → Dashboard 收到 → broadcast 到所有浏览器终端
      → xterm.js 渲染
```

## 连接管理

- 同一时间只有一个 VM 连接
- 新 VM 连接时，旧连接被关闭
- 浏览器终端随时可以连接/断开，不影响 VM 连接
- VM 断线后，浏览器终端继续正常使用 bash

## stdin (规划中)

```
浏览器输入 → Dashboard → WebSocket send to VM → VM 读取
```
