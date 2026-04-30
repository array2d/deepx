# dashboard 前端交互方案

> Web 可视化面板。React/TS 前端 → Go 后端 → exec loader 二进制。
> 复用 loader 解析 dxlang，dashboard 不重新实现语法解析。

## 架构

```
Browser (React/TS) ──HTTP──→ Go Backend ──exec──→ loader (写 /src/func/*)
       │                         │                        │
       │                    GET/POST Redis              Redis KV Space
       │                         │                        │
       └─── WebSocket ──── status/vthread push ──────────┘
```

## 目录结构

```
tool/dashboard/
├── CLAUDE.md
├── go.mod / go.sum
├── main.go                    Go 入口
├── internal/
│   ├── server/server.go       HTTP + WS 路由 + 静态文件服务
│   ├── handler/
│   │   ├── status.go          /api/status + /ws/status
│   │   └── run.go             /api/run (源码→临时文件→exec loader→vthread)
│   └── redis/client.go        Redis 连接 + 状态读取 + vthread 管理
└── frontend/                  React + TypeScript (Vite)
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── App.css
        ├── components/
        │   ├── StatusPanel.tsx    左栏：组件状态实时监控
        │   ├── CodeEditor.tsx     中栏：dxlang 代码编辑 + Run
        │   ├── OutputPanel.tsx    右栏：执行结果 / VThread 列表
        │   └── VThreadRow.tsx     vthread 单行展开
        ├── hooks/
        │   ├── useWebSocket.ts    WebSocket 连接管理
        │   └── useVThreadPoll.ts  vthread 状态轮询
        └── api/
            └── client.ts          HTTP API 调用封装
```

## 前端三栏布局

```
┌──────────────────────────────────────────────────────┐
│  deepx dashboard              redis: 127.0.0.1:16379 │
├──────────┬──────────────────────────────┬────────────┤
│ Status   │  Code Editor                 │  Output    │
│ Panel    │                              │  Panel     │
│          │  1 def add(A:int, B:int)     │            │
│ ● op-plat│  2     A + B -> ./C          │  vtid=42   │
│ ● heap   │  3                           │  done 1.2s │
│ ● vm     │  4 def main() -> (Z:int) {   │            │
│          │  5     add(3,5) -> ./Z       │  VThreads  │
│ funcs: 3 │  6 }                         │  ────────  │
│ vthread:2│                              │  42 done   │
│          │  [▶ Run] [Check] [Clear]     │  43 run    │
└──────────┴──────────────────────────────┴────────────┘
```

## API 设计

| Method | Path | 说明 |
|--------|------|------|
| GET | / | SPA index.html (开发时 proxy 到 Vite) |
| GET | /api/status | 组件状态快照 |
| WS | /ws/status | 状态实时推送 (2s) |
| POST | /api/run | 提交 dxlang 代码执行 |
| GET | /api/vthread/:id | 查询 vthread 状态 |
| GET | /api/vthreads | 列出活跃 vthread |
| GET | /api/functions | 列出已注册函数 |

## run 流程

```
POST /api/run { source, entry?, timeout }
  │
  ├─ 1. 写源码到临时文件 /tmp/dashboard-XXXX.dx
  ├─ 2. exec loader /tmp/dashboard-XXXX.dx <redis_addr>
  ├─ 3. 解析 loader 输出 → 函数列表 + entry 信息
  ├─ 4. INCR /sys/vtid_counter → vtid
  ├─ 5. Pipeline SET vthread (status=init, pc=[0,0], opcode=entryFunc)
  ├─ 6. LPUSH notify:vm
  └─ 7. 返回 { vtid, status: "init" }
       │
       ▼ (前端轮询或 WS 订阅)
  GET /api/vthread/:vtid → { status, pc, error?, duration }
```

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | React 18 + TypeScript + Vite |
| 样式 | CSS (暗色主题, 类终端风格) |
| 后端 | Go + net/http + gorilla/websocket |
| Redis | go-redis/v9 |
| 部署 | Go embed 打包前端 build 产物 → 单二进制 |
