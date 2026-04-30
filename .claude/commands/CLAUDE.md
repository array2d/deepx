# 项目命令

> **所有构建命令均通过根目录 `Makefile` 执行。禁止使用裸 `go build`、`cmake`、`./build.sh`。**

## 构建命令

| 命令 | 组件 | 语言 | 说明 |
|------|------|------|------|
| `/build-op-cuda` | op-plat (CUDA) | C++/CUDA | 构建 CUDA 计算平面 |
| `/build-op-metal` | op-plat (Metal) | C++/ObjC++/Metal | → `make build-op-metal` |
| `/build-heap-metal` | heap-plat (Metal) | C++/ObjC++ | → `make build-heap-metal` |
| `/build-vm` | VM | Go | → `make build-vm` |
| `/build-all` | 全部 | C++/Go | → `make build-all` |

## 测试命令

| 命令 | 说明 |
|------|------|
| `/test-vm` | 运行 VM 单元测试 → `make test-vm` |
| `/test-integration` | 运行 VM 集成测试 → `make test-integration` |
| `/test-real` | 运行真实联调测试 → `make test-real` |
| `/test-op-metal` | 构建 op-metal 并运行 shm 跨进程测试 |

## 服务生命周期 (联调)

| 命令 | 说明 |
|------|------|
| `/start-services` | 后台启动 heap-metal + op-metal → `make start-services` |
| `/stop-services` | 停止 heap-metal + op-metal 进程 → `make stop-services` |
| `/status` | 查看所有服务状态 → `make status` |
| `/pipeline` | 完整联调流水线 → `make pipeline` |

## 环境命令

| 命令 | 说明 |
|------|------|
| `/reset-redis` | 重置 Redis → `make reset-redis` |

## 联调命令详解

### 完整流水线 (`/pipeline`)
```bash
make pipeline REDIS_ADDR=127.0.0.1:16379
```
等价于依次执行:
1. `make build-all`      — 编译所有项目
2. `make start-services` — 后台启动两个二进制
3. `make reset-redis`    — 清空 Redis
4. `make test-real`      — 运行 Go 真实联调测试
5. `make stop-services`  — 停止后台进程

### 分步联调

```bash
# 所有命令从项目根目录执行

# 1. 构建
make build-all REDIS_ADDR=127.0.0.1:16379

# 2. 启动后台服务
make start-services REDIS_ADDR=127.0.0.1:16379
make status REDIS_ADDR=127.0.0.1:16379  # 检查状态

# 3. 重置 Redis
make reset-redis REDIS_ADDR=127.0.0.1:16379

# 4. 运行联调测试
make test-real REDIS_ADDR=127.0.0.1:16379

# 5. 停止服务
make stop-services

# 查看日志 (通过 executor/Makefile)
cd executor && make logs-op    # op-metal 日志
cd executor && make logs-heap  # heap-metal 日志
```

### 进程架构

```
Go test (integration_real_test.go)
  ├── exec: heap-metal (subprocess)  → 消费 cmd:heap-metal:0
  │     └── 注册: /sys/heap-plat/heap-metal:0
  ├── exec: op-metal   (subprocess)  → 消费 cmd:op-metal:0
  │     └── 注册: /sys/op-plat/op-metal:0
  └── goroutine: engine.RunWorker    → 消费 notify:vm, 推进 vthread
```

测试流程:
1. `startPlat()` 启动二进制子进程，阻塞等待 Redis 注册
2. `engine.RunWorker()` 在 goroutine 中消费 vthread
3. 加载 dxlang 源码，创建 vthread，推送到 notify:vm
4. VM → push cmd:heap-metal:0 / cmd:op-metal:0
5. heap/op 执行完成 → push done:<vtid>
6. `waitVthreadDone()` 轮询 vthread 状态，验证结果
7. `t.Cleanup()` 自动 kill 子进程
