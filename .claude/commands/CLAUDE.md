# 项目命令

> **所有构建命令均通过根目录 `Makefile` 执行。禁止使用裸 `go build`、`cmake`、`./build.sh`。**

## 构建命令

| 命令 | 组件 | 语言 | 说明 |
|------|------|------|------|
| `/build-op-cuda` | op-plat (CUDA) | C++/CUDA | 构建 CUDA 计算平面 |
| `/build-exop-metal` | op-plat (Metal) | C++/ObjC++/Metal | → `make build-exop-metal` |
| `/build-heap-metal` | heap-plat (Metal) | C++/ObjC++ | → `make build-heap-metal` |
| `/build-io-metal` | io-plat | C++ | → `make build-io-metal` |
| `/build-vm` | VM | Go | → `make build-vm` |
| `/build-all` | 全部 | C++/Go | → `make build-all` |

## 测试命令

| 命令 | 说明 |
|------|------|
| `/test-vm` | 运行 VM 单元测试 → `make test-vm` |
| `/test-integration` | 运行 VM 集成测试 → `make test-integration` |
| `/test-exop-metal` | 构建 exop-metal 并运行 shm 跨进程测试 |

## 服务生命周期 (联调)

| 命令 | 说明 |
|------|------|
| `/boot` | 构建并启动所有服务 (deepxctl boot) |
| `/run <file.dx>` | 加载 .dx 文件 (如有顶层调用则自动执行) → deepxctl run |
| `/run --rm <file.dx>` | 执行后自动 shutdown + FLUSHDB，适批量测试 |
| `/run --entry <f> <file.dx>` | 手动指定入口函数执行 |
| `/shutdown` | 停止所有 booted 服务 (deepxctl shutdown) |
| `/status` | 查看所有服务状态 → `make status` |
| `/pipeline` | 完整联调流水线 → `make pipeline` |

## 环境命令

| 命令 | 说明 |
|------|------|
| `/reset-redis` | 重置 Redis → `make reset-redis` |
| `/kvspace <vtid>` | 检查 KV Space 联调状态 (堆 + 栈) → skill `debug-kvspace` |

## 联调命令详解

### 典型联调流程 (deepxctl)

```bash
# 1. 构建并启动所有服务 (一次性)
deepxctl boot

# 2a. 加载纯定义文件 (只定义函数，不执行)
deepxctl run example/dxlang/builtin/call/add_test.dx
#  → "Loaded 1 function(s) into KV Space."
#  → 没有顶层调用，VM 的 /func/main 监视器保持等待

# 2b. 加载含顶层调用的文件 (自动执行)
deepxctl run my_file_with_call.dx
#  → loader 检测到顶层调用 → 写入 /func/main → VM 自动创建 vthread → 执行

# 2c. 手动指定入口函数 (builtin 标量运算，仅需 VM)
deepxctl run --entry native_arith example/dxlang/builtin/native/arith/add.dx
#  → 绕过顶层调用检测，直接以 native_arith 为入口执行

# 3. 停止所有服务
deepxctl shutdown
```

### 批量测试 (`--rm`)

`--rm` 执行完自动 shutdown + FLUSHDB，每次跑完环境归零，适合批量跑多个 .dx 文件：

```bash
# 批量测试 builtin 算子
for f in example/dxlang/builtin/native/arith/*.dx; do
    echo "=== $f ==="
    deepxctl run --rm "$f"
done
```

### 分步联调 (make)

```bash
# 构建
make build-all

# 启动服务 (后台，通过 PID 文件管理)
make start-services REDIS_ADDR=127.0.0.1:16379
make status REDIS_ADDR=127.0.0.1:16379

# 重置 Redis
make reset-redis REDIS_ADDR=127.0.0.1:16379

# 加载 & 执行 (tensor 示例，需 VM + heap-plat + op-plat)
./tmp/deepx-vm/loader load example/dxlang/tensor/lifecycle/compute.dx
# 或 builtin 示例 (仅需 VM)
./tmp/deepx-vm/loader load example/dxlang/builtin/native/arith/add.dx
./tmp/deepx-vm/vm 127.0.0.1:16379  # 手动启动 VM

# 停止服务
make stop-services

# 查看日志
tail -f /tmp/deepx-logs/exop-metal.log
tail -f /tmp/deepx-logs/heap-metal.log
tail -f /tmp/deepx-logs/io-metal.log
tail -f /tmp/deepx-logs/vm.log
```

### plats 管理

plats (exop-metal / heap-metal / io-metal / VM) 由 deepxctl 通过 subprocess 管理生命周期，**所有通信严格通过 Redis**:
- `deepxctl boot` 启动所有进程，写入 PID 到 `/tmp/deepx-boot.json`
- `deepxctl run` 检测 boot 状态后加载并执行 .dx 文件
- `deepxctl shutdown` **有序退出**: plats → VM → 心跳验证 → deepxctl 退出

**Redis 队列分工**：
| 队列 | 类型 | 用途 |
|------|------|------|
| `cmd:exop-metal:0` | 业务 | GPU 计算指令 |
| `cmd:heap-metal:0` | 业务 | 堆内存管理指令 |
| `cmd:io-metal:0` | 业务 | I/O 指令 (print/save/load) |
| `notify:vm` | 业务 | VThread 调度通知 |
| `/func/main` | **入口** | loader→VM→deepxctl 三方协作: `{"entry":"f","reads":[...],"writes":[...]}` → `{"vtid":"...","status":"executing"}` → `{"vtid":"...","status":"done"}` |
| `sys:cmd:exop-metal:0` | 系统 | op-metal shutdown |
| `sys:cmd:heap-metal:0` | 系统 | heap-metal shutdown |
| `sys:cmd:io-metal:0` | 系统 | io-metal shutdown |
| `sys:cmd:vm:0` | 系统 | VM shutdown |
| `/sys/heartbeat/*` | **心跳** | 各组件每 2s 上报 `{"ts":...,"status":"running/stopped"}` |

**退出顺序**: plats (exop-metal, heap-metal, io-metal) 先退出 → VM 退出 → deepxctl 检查心跳确认全部 stopped → 清理 PID 文件
OS SIGKILL 仅作为 Redis 不可达时 / 超时时的最后兜底。

参见 `tool/deepxctl/internal/process/manager.go` (进程管理)、`tool/deepxctl/cmd/shutdown.go` (有序 shutdown + 心跳验证) 和各组件 main 文件 (心跳上报 + 系统指令监听)。