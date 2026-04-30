# 项目命令

## 构建命令

| 命令 | 组件 | 语言 | 说明 |
|------|------|------|------|
| `/build-op-cuda` | op-plat (CUDA) | C++/CUDA | 构建 CUDA 计算平面 |
| `/build-op-metal` | op-plat (Metal) | C++/ObjC++/Metal | 构建 Metal 计算平面 |
| `/build-heap-metal` | heap-plat (Metal) | C++/ObjC++ | 构建 Metal 堆管理平面 |
| `/build-vm` | VM | Go | 构建 VM (含单元测试 + 编译) |
| `/build-all` | 全部 | C++/Go | 构建 op-metal + heap-metal + VM |

## 测试命令

| 命令 | 说明 |
|------|------|
| `/test-vm` | 运行 VM 单元测试 (不包括集成测试) |
| `/test-integration` | 运行 VM 集成测试 (需要 Redis，仅纯 VM 算子) |
| `/test-real` | 运行真实联调测试 (需要 Redis + heap-metal + op-metal 二进制) |
| `/test-op-metal` | 构建 op-metal 并运行 shm 跨进程测试 |

## 服务生命周期 (联调)

| 命令 | 说明 |
|------|------|
| `/start-services` | 后台启动 heap-metal + op-metal，日志写入 /tmp/deepx-logs/ |
| `/stop-services` | 停止 heap-metal + op-metal 进程 |
| `/status` | 查看所有服务状态 (Redis / heap / op) |
| `/pipeline` | 完整联调流水线: build → start → reset-redis → test-real → stop |

## 环境命令

| 命令 | 说明 |
|------|------|
| `/reset-redis` | 重置 Redis（FLUSHDB，清理所有 vthread/src/op/sys/done/cmd key） |

## 联调命令详解

### 完整流水线 (`/pipeline`)
```bash
cd executor && make pipeline REDIS_ADDR=127.0.0.1:16379
```
等价于依次执行:
1. `make build`        — 编译 op-metal + heap-metal
2. `make start-services` — 后台启动两个二进制
3. `make reset-redis`  — 清空 Redis
4. `make test-real`    — 运行 Go 真实联调测试
5. `make stop-services` — 停止后台进程

### 分步联调

```bash
# 1. 构建
make build REDIS_ADDR=127.0.0.1:16379

# 2. 启动后台服务
make start-services REDIS_ADDR=127.0.0.1:16379
make status  # 检查状态

# 3. 重置 Redis
make reset-redis REDIS_ADDR=127.0.0.1:16379

# 4. 运行联调测试
make test-real REDIS_ADDR=127.0.0.1:16379

# 5. 停止服务
make stop-services

# 查看日志
make logs-op    # op-metal 日志
make logs-heap  # heap-metal 日志
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
