# VM 对接 dashboard 终端 I/O 方案

## 概述

Dashboard 的 Web 终端通过 WebSocket 连接到浏览器。Dashboard 启动时将终端注册到 Redis KV space，VM 通过 Redis 将输出转发到终端。

```
浏览器 xterm.js ←WebSocket→ Dashboard ←Redis Pub/Sub← VM print/cerr
```

## 注册位置

Dashboard 启动时写入：

```
/sys/term/dashboard/stdout  → HASH {type: "websocket", detail: "term:dashboard:stdout"}
/sys/term/dashboard/stderr  → HASH {type: "websocket", detail: "term:dashboard:stderr"}
/sys/term/dashboard/stdin   → HASH {type: "websocket", detail: "term:dashboard:stdin"}
```

每个 key 是 Redis HASH：
- `type`: `"websocket"` — 终端侧通过 WebSocket 连接到浏览器
- `detail`: Redis channel 名称 — VM 通过 `PUBLISH` 把数据发到这个 channel

## type 含义

| type 值       | 含义                         | VM 操作                      |
|---------------|------------------------------|------------------------------|
| `"websocket"` | 终端通过 WebSocket 连接浏览器 | `PUBLISH <detail> <message>` |

## VM 对接步骤

### 1. 发现终端 channel

```
KEYS /sys/term/*/stdout
KEYS /sys/term/*/stderr
```

### 2. 读取 HASH

```
TYPE /sys/term/dashboard/stdout       → "hash"
HGET /sys/term/dashboard/stdout type  → "websocket"
HGET /sys/term/dashboard/stdout detail → "term:dashboard:stdout"
```

### 3. 发布消息

```
PUBLISH term:dashboard:stdout "hello"
PUBLISH term:dashboard:stderr "error"
```

### 4. 数据流

```
VM PUBLISH term:dashboard:stdout "hello"
  → Dashboard SUBSCRIBE (常驻)
    → WebSocket → xterm.js 渲染
```

## 兼容

- `/sys/term/default/stdout` 旧格式（`file://...`）作为 fallback 保留
- VM 同时支持 HASH（`type: "websocket"`）和 string（`file://`）

## 伪代码

```go
func writeTerm(rdb, stream, line) {
    keys := rdb.Keys("/sys/term/*/" + stream)
    for _, key := range keys {
        if rdb.Type(key) == "hash" {
            if rdb.HGet(key, "type") == "websocket" {
                ch := rdb.HGet(key, "detail")
                rdb.Publish(ch, line)
            }
        }
    }
    // fallback: 写 file:// 到 /sys/term/default/stdout
}
```

## stdin (规划中)

```
VM 请求输入 → PUBLISH term:dashboard:stdin "?<vtid>"
   → Dashboard 转发到浏览器 → 用户输入 → 浏览器 → Dashboard
     → PUBLISH term:dashboard:stdin:resp "<vtid>:<input>"
       → VM 订阅获取
```
