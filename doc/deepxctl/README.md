# deepxctl run — 一键运行 dx 代码

> `deepxctl run xxx.dx` 一条命令完成 Redis 准备、组件启动、代码加载、执行、收尾。
> 职责边界见 [CLAUDE.md](CLAUDE.md)。

---

## 1. 解决的问题

当前调试 .dx 代码需手动 8 步：

```
redis-server --port 16379 &
redis-cli -p 16379 FLUSHDB
./executor/op-metal/build.sh
./executor/heap-metal/build.sh
./executor/vm/build.sh
/tmp/deepx/op-metal/build/deepx-op-metal 127.0.0.1 16379 &
/tmp/deepx/heap-metal/build/deepx-heap-metal 127.0.0.1 16379 &
VM_ID=0 /tmp/deepx-vm/vm 127.0.0.1:16379 &
/tmp/deepx-vm/loader example/dxlang/tensor/lifecycle/compute.dx 127.0.0.1:16379
# 还需要手动创建 vthread...
```

deepxctl 的目标：

```
deepxctl run example/dxlang/tensor/lifecycle/compute.dx
```

---

## 2. 执行流程

```
deepxctl run full.dx
│
├─ [1/6] Redis
│         PING → 不可达则报错退出
│         FLUSHDB → 重置 KV 空间
│
├─ [2/6] Build (按需)
│         检测二进制是否存在:
│           /tmp/deepx/op-metal/build/deepx-op-metal
│           /tmp/deepx/heap-metal/build/deepx-heap-metal
│           /tmp/deepx-vm/vm
│           /tmp/deepx-vm/loader
│         缺失 → exec build.sh 构建
│
├─ [3/6] 启动子进程（按依赖顺序）
│         ① op-plat   → GET /sys/op-plat/metal:0 等待 status=running
│         ② heap-plat → GET /sys/heap-plat/metal:0 等待 status=running
│         ③ VM        → GET /sys/vm/0 等待 status=running
│
├─ [4/6] 加载 dx 代码
│         exec loader 二进制 → 写入 /src/func/<name>
│         验证: GET /src/func/<name> 非空
│
├─ [5/6] 创建 vthread + 执行
│         INCR /sys/vtid_counter → vtid
│         SET /vthread/<vtid> = {"pc":"[0,0]","status":"init"}
│         SET /vthread/<vtid>/[0,0] = "<entry_func>"
│         SET /vthread/<vtid>/[0,1] = "./ret"
│         LPUSH notify:vm {"event":"new_vthread","vtid":"<vtid>"}
│         ┌─ 轮询 GET /vthread/<vtid>
│         │   status=done  → 成功
│         │   status=error → 打印错误
│         └─ 超时 → TIMEOUT_ERROR
│
└─ [6/6] 清理
          SIGTERM → 等待 2s → SIGKILL 所有子进程
```

---

## 3. 入口函数确定规则

deepxctl 创建 vthread 只写**一条顶层 CALL 指令**：

```
/vthread/<vtid>/[0,0] = "<entry_func>"    # CALL 操作码
/vthread/<vtid>/[0,1] = "./ret"           # 返回值槽位
```

VM 拾取后自动识别 `<entry_func>` 非内置关键字 → 查找 `/src/func/<entry_func>` → CALL eager 翻译 → 展开函数体到子栈。

**入口函数名规则**：
1. 文件中有 `def main` → 用 `main`
2. 只有一个 `def` → 用那个名字
3. 多个 `def` 且无 `main` → 报错，要求 `--entry` 指定

入口函数名从 loader 的输出获取（loader 加载时打印 `→ /src/func/<name>`），或 GET `/src/func/*` KEYS 推断。

---

## 4. CLI 用法

```
deepxctl run [flags] <file.dx>

flags (必须放在 .dx 文件路径之前):
  -r, --redis string    Redis 地址 (默认: 127.0.0.1:16379)
      --rm              执行后自动 shutdown + FLUSHDB，保证所有组件关闭
      --entry string    指定入口函数名 (多 def 且无 main 时必须)
      --timeout int     执行超时秒数 (默认: 60, 0=无限制)
      --boot            未 boot 时自动 boot (默认: true)
      --boot=false      不自动 boot，未 boot 则报错

示例:
  deepxctl run example/dxlang/tensor/lifecycle/compute.dx
  deepxctl run -v example/dxlang/tensor/call/tensor_pipeline.dx
  deepxctl run --entry stage1 --timeout 30 example/dxlang/tensor/call/tensor_pipeline.dx
  deepxctl run --rm example/dxlang/tensor/lifecycle/compute.dx   # 一键执行+清理
```

---

## 5. 输出格式

```
$ deepxctl run example/dxlang/tensor/lifecycle/compute.dx

 deepxctl  |  redis: 127.0.0.1:16379
─────────────────────────────────────────

[1/6] Redis ........................ ✓
[2/6] Build ........................ ✓ (up-to-date)
[3/6] op-plat ...................... ✓ (pid=12345)
      heap-plat .................... ✓ (pid=12346)
      VM ........................... ✓ (pid=12347)
[4/6] Load: full.dx ................ ✓ (/src/func/lifecycle_full)
[5/6] Execute: lifecycle_full ...... ✓ (vtid=1, done, 0.042s)
[6/6] Cleanup ...................... ✓

─────────────────────────────────────────
SUCCESS  vtid=1  status=done  42ms
─────────────────────────────────────────
```

错误输出：

```
$ deepxctl run bad.dx

 deepxctl  |  redis: 127.0.0.1:16379
─────────────────────────────────────────

[1/6] Redis ........................ ✓
[2/6] Build ........................ ✓
[3/6] op-plat ...................... ✓
      heap-plat .................... ✓
      VM ........................... ✓
[4/6] Load: bad.dx ................. ✗

─────────────────────────────────────────
ERROR  loader exit code 1
  /src/func/ not found after loading bad.dx
─────────────────────────────────────────
```

---

## 6. 错误码

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 10 | Redis 连接失败 |
| 20 | 组件构建失败 |
| 30 | 子进程启动失败/超时 |
| 40 | dx 加载失败 |
| 50 | vthread 执行失败 (status=error) |
| 60 | vthread 执行超时 |
| 99 | 内部错误 |

---

## 7. 实现结构

```
tool/deepxctl/                  # 复用现有目录
├── main.go                     # 入口
├── cmd/run.go                  # run 子命令
├── internal/
│   ├── redis/redis.go          # 连接 + FLUSHDB
│   ├── build/builder.go        # exec build.sh 子进程
│   ├── process/manager.go      # 子进程启动/停止/就绪等待
│   ├── loader/loader.go        # exec loader 子进程
│   └── executor/executor.go    # vthread 创建 + 轮询
└── tensor/                     # 已有，不动
```

---

## 8. 与组件的关系

deepxctl 只做**进程编排**，不实现任何计算逻辑：

```
deepxctl                   组件
────────                  ─────
✓ 启动/停止子进程         ✗ 不实现 tensor 计算 (op-plat)
✓ 调用 build.sh           ✗ 不分配 shm (heap-plat)
✓ 调用 loader             ✗ 不翻译 dxlang (VM)
✓ FLUSHDB                 ✗ 不生产 cmd:* 队列消息 (VM)
✓ 写 /vthread/<vtid> init ✗ 不解析 dxlang 语法 (loader/VM)
✓ 轮询 status             ✗ 不消费 done:* 队列 (VM)
✓ 结束清理子进程          ✗ 不注册算子 (op-plat 自行注册)
```

详细约束见 [CLAUDE.md](CLAUDE.md)。
