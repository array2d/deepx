# deepx

> **deepx = Redis KV 空间 + 7 核协作（pysdk 写源码、heap-plat 管堆、op-plat 管算、io-plat 管 I/O、VM 管执行、deepxctl 管编排、deepx-core 统一类型契约），由 deepx-core 类型系统统一契约。**

## 架构铁律

> **拒绝包庇垃圾架构。为了架构的简洁、优美和唯一性，可以耗费无限的时间和精力来修改一切。**

## 组件职责

### 公共核心库 (deepx-core)

> **deepx-core** 是 deepx 所有 C++ 组件的统一公共库——提供跨平台的类型系统(dtype)、张量类型(Shape/Tensor)、共享内存(shmem)、注册接口(Registry)与工具类(stdutil)。不绑定任何特定硬件后端。

| 组件 | 一句话职责 |
|------|-----------|
| **executor/deepx-core** | 平台无关的 C++ 静态库（libdeepx_core.a），统一 dtype/tensor/shmem/registry/stdutil，被所有 executor 依赖 |

### 语言层

> **dxlang** 是 deepx 元程级的编程语言——定义统一的类型系统与可序列化协议，是前端/编译器/调度器/执行器之间的契约。

| 组件 | 一句话职责 |
|------|-----------|
| **executor/dxlang** | ⚠️ 代码已迁移至 deepx-core，目录保留兼容（CMake target `deepxcore` 将被 `deepx_core` 替代） |
| **executor/common-metal** | Metal HAL 库：封装 Metal 设备查询（`metal_device`）。shm_tensor/registry 已迁移至 deepx-core |

### 堆平面 (heap-plat)

| 组件 | 一句话职责 |
|------|-----------|
| **heap-metal** | 进程维持 deepx 元程的堆在 Metal 设备平台的高可用——管理 tensor 的 shm 创建/引用/删除 |
| **heap-cpu** | 进程维持 deepx 元程的堆在 CPU 平台的高可用——管理 tensor 的 shm 创建/引用/删除 |
| **heap-cuda** | （待开发）进程维持 deepx 元程的堆在 CUDA 设备平台的高可用 |

### 扩展算子平面 (exop-plat)

| 组件 | 一句话职责 |
|------|-----------|
| **exop-cpu** | 被动消费 `cmd:op-cpu:*`，在 CPU 上以 OpenMP + SIMD 执行张量运算 |
| **op-metal** | 被动消费 `cmd:op-metal:*`，在 Apple GPU 上执行张量计算，完成后通知 `done:<vtid>` |
| **op-cuda** | 被动消费 `cmd:op-cuda:*`，在 NVIDIA GPU 上执行张量计算，完成后通知 `done:<vtid>` |

### I/O 平面 (io-plat)

| 组件 | 一句话职责 |
|------|-----------|
| **io-metal** | 被动消费 `cmd:io-metal:*`，执行 tensor 的 print/save/load 等 I/O 操作，完成后通知 `done:<vtid>` |

### 虚拟机

| 组件 | 一句话职责 |
|------|-----------|
| **vm** | 元程虚拟机：CALL 时 eager 翻译源码→执行层坐标、路由指令到 op-plat/heap-plat、推进 vthread 状态、本地求值原生算子 |

### 编排器

| 组件 | 一句话职责 |
|------|-----------|
| **deepxctl** | deepx 命令行编排器：`boot` 启动服务 → `run` 执行 .dx → `shutdown` 停止服务，三步分离职责 |

### 前端

| 组件 | 一句话职责 |
|------|-----------|
| **front/go** | Go 语言深度学习模块库，提供 tensor 运算、神经网络层、transformer 等 API |
| **front/py** | Python 算法前端，提供 tensor 运算、神经网络模块、优化器 API，并将 dxlang 源码注册到 KV 空间 |

### 模型工具

| 组件 | 一句话职责 |
|------|-----------|
| **model/h5_deepx** | HDF5 格式深度学习模型的加载、转换与导出工具 |
| **model/onnx_deepx** | ONNX 格式深度学习模型的加载、转换与导出工具 |

### 遗留

| 组件 | 一句话职责 |
|------|-----------|
| **old-cppcommon** | 旧版 C++ Tensor/Shape/TF 框架，核心类型已迁移至 deepx-core，保留算子接口(tensorfunc)和 TF 框架供 exop-cpu 旧架构使用 |
| **op-mem-ompsimd** | ⚠️ 已废弃，由 exop-cpu 替代 |

## C++ 组件依赖关系

```
                    ┌─────────────────────────────────┐
                    │         deepx-core (STATIC)       │
                    │  dtype / tensor / shmem /         │
                    │  registry / stdutil               │
                    └──────┬──────────┬─────────────────┘
                           │          │
              ┌────────────┼──────────┼──────────────┐
              │            │          │              │
              ▼            ▼          ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ exop-cpu │ │ op-cuda  │ │ op-metal │ │heap-metal│
        │  (CPU)   │ │ (CUDA)  │ │ (Metal)  │ │ (Metal)  │
        └──────────┘ └──────────┘ └────┬─────┘ └────┬─────┘
                                       │            │
                                       ▼            ▼
                                ┌────────────────────────┐
                                │ common-metal (HAL only) │
                                │  metal_device           │
                                └────────────────────────┘
```

**依赖说明**:
- `deepx-core` — 所有 executor 的共同依赖（平台无关）
- `common-metal` — 仅 op-metal / heap-metal 需要（Apple Metal HAL）
- `exop-cpu` / `op-cuda` / `heap-cpu` — 仅依赖 deepx-core，无需 Metal
- `old-cppcommon/tensorfunc` — 算子接口定义（Dispatcher 模式），多后端可复用
- `old-cppcommon/tf` — TF 框架（仅 exop-cpu 旧架构依赖，逐步淘汰）

## 文档

| 目录 | 用途 | 文件 |
|------|------|------|
| `doc/metaproc/` | 整体架构、Redis key、开发指南、速度赶超策略 | `spec-v1.md` `deepx-design.md` `redis-keys.md` `deepx-speed-strategy.md` `dev-heap-plat.md` `dev-op-plat.md` `dev-pysdk.md` `dev-vm.md` |
| `doc/vm/` | VM 设计、调度器 | `README.md` `scheduler.md` |
| `doc/dxlang/` | dxlang 语言设计（类型系统、控制流、编译器分析） | `README.md` `compiler-analysis-ssa-vs-arrow.md` `control-flow.md` |
| `doc/heap-plat/` | 堆管理平面 (tensor 生命周期) | `README.md` `heap-metal.md` `heap-cuda.md` `heap-cpu.md` |
| `doc/op-plat/` | 计算平面 (算子注册、GPU kernel) | `README.md` `op-metal.md` `op-cuda.md` `op-cpu.md` |
| `doc/plans/` | 架构演进规划 | `cpp-core-libs-integration.md` |

按任务查阅: 架构→`metaproc/` · dxlang→`dxlang/` · op开发→`op-plat/` · heap开发→`heap-plat/` · VM开发→`vm/` · 核心库→`deepx-core/`

## 术语

见 `.claude/glossary.md`（元程核心术语速查）。

## 构建 / 测试

> **强制规则: 所有构建必须通过根目录 `make` 命令执行，禁止直接使用 `go build`、`cmake`、`./build.sh` 等裸命令。**

### 构建

| 命令 | 说明 |
|------|------|
| `make build-all` | 构建全部项目 (VM + deepxctl + dashboard + op-metal + heap-metal + exop-cpu + heap-cpu + io-metal) |
| `make build-vm` | 构建 VM + loader (Go) → `/tmp/deepx-vm/vm` `/tmp/deepx-vm/loader` |
| `make build-deepxctl` | 构建 deepxctl CLI (Go) → `/tmp/deepx-vm/deepxctl` |
| `make build-dashboard` | 构建 dashboard (Go+React) → `/tmp/deepx-dashboard/` |
| `make build-op-metal` | 构建 Metal 计算平面 (C++/Metal cmake) → `/tmp/deepx/op-metal/build/deepx-op-metal` |
| `make build-heap-metal` | 构建 Metal 堆管理平面 (C++ cmake) → `/tmp/deepx/heap-metal/build/deepx-heap-metal` |
| `make build-exop-cpu` | 构建 CPU 扩展算子平面 (C++ cmake) → `/tmp/deepx/exop-cpu/build/deepx-exop-cpu` |
| `make build-heap-cpu` | 构建 CPU 堆管理平面 (C++ cmake) → `/tmp/deepx/heap-cpu/build/deepx-heap-cpu` |
| `make build-io-metal` | 构建 I/O 平面 (C++ cmake) → `/tmp/deepx/io-metal/build/deepx-io-metal` |

### 测试

| 命令 | 说明 |
|------|------|
| `make test-vm` | 运行 VM 单元测试 |
| `make test-integration` | 运行 VM 集成测试 (需要 Redis，纯 VM 算子) |
| `/test-op-metal` | 构建 op-metal 并运行 shm 跨进程测试 (C++ 独立测试) |
| `make pipeline` | 完整联调流水线: build → start-services → reset-redis → stop |
| `make reset-redis` | 重置 Redis 测试环境 (FLUSHDB) |

### deepxctl 联调架构

deepxctl 将生命周期拆分为三个独立命令，**所有组件间通信严格通过 Redis KV Space**：

```bash
deepxctl boot       # 构建 + 启动 op-metal、heap-metal、VM，写入 PID 文件 /tmp/deepx-boot.json
deepxctl run a.dx   # 检测 boot 状态 → loader 加载 dx → 自动检测 /func/main → 等待结果 (可多次执行)
                    #   --rm: 执行后自动 FLUSHDB + shutdown (一键清理)
                    #   --entry <name>: 手动指定入口函数 (即使文件无顶层调用也会执行)
deepxctl shutdown   # 有序退出: plats → VM → 心跳验证 → 清理
```

**dxlang 执行语义 (v2)**：
- **纯定义文件** (只有 `def` 块，无顶层调用): `deepxctl run` 仅加载函数定义到 `/src/func/*`，不执行任何代码。VM 的 `/func/main` 监视器保持等待。
- **包含顶层调用的文件** (在 `def { }` 块外部有 `funcName(args) -> outputs`): loader 自动写入 `/func/main`，VM 检测后创建 vthread 并执行。deepxctl 轮询等待结果。
- **手动指定入口**: `--entry <funcName>` 绕过顶层调用检测，直接写入 `/func/main`。
- 关键 Redis key: `/func/main` — 入口协议 (`{"entry":"funcName","reads":[...],"writes":[...]}` → VM 认领后改为 `{"vtid":"...","status":"executing"}` → 最终 `{"vtid":"...","status":"done"}`)

**通信规则**：
- 业务队列: `cmd:op-metal:0`, `cmd:heap-metal:0`, `notify:vm`
- 系统队列 (`sys:` 前缀): `sys:cmd:op-metal:0`, `sys:cmd:heap-metal:0`, `sys:cmd:vm:0`
- 入口协议: `/func/main` (loader → VM → deepxctl 三方协作)
- 心跳上报: `/sys/heartbeat/op-metal:0`, `/sys/heartbeat/heap-metal:0`, `/sys/heartbeat/vm:0`
  - 各组件每 2s 上报 `{"ts":...,"status":"running","pid":...}`
  - 退出时上报 `{"status":"stopped"}` — shutdown 以此验证退出完成
- **严禁跨组件 OS 信号** — shutdown 通过 Redis `sys:cmd:*` 发送 `{"cmd":"shutdown"}` 触发各组件优雅退出
- **退出顺序**: plats (op-metal, heap-metal) → VM → 心跳验证 → deepxctl 退出
- OS SIGKILL 仅作为 Redis 不可达时 / 超时时的最后兜底

进程管理: `tool/deepxctl/internal/process/manager.go`
boot/run/shutdown 逻辑: `tool/deepxctl/cmd/{boot,run,shutdown}.go`
心跳: 各组件 main 文件 (每 2s SET `/sys/heartbeat/*`)
日志文件: `/tmp/deepx-logs/{op-metal,heap-metal,vm}.log`

### 加载示例到 KV Space

```bash
# 构建后 loader 位于 /tmp/deepx-vm/loader

# 加载单个 tensor 文件
./tmp/deepx-vm/loader load example/dxlang/tensor/lifecycle/compute.dx

# 加载整个 tensor 目录
./tmp/deepx-vm/loader load example/dxlang/tensor/nn/

# 加载 builtin 目录 (VM 标量运算，无需 GPU 后端)
./tmp/deepx-vm/loader load example/dxlang/builtin/

# 列出已注册函数
./tmp/deepx-vm/loader ls

# 加载并执行 builtin 函数 (仅需 VM，无需 heap-plat/op-plat)
./tmp/deepx-vm/loader run example/dxlang/builtin/native/arith/add.dx native_arith "./a:2" "./b:3" -- "./c"
```

## 开发 Agents

| Agent | 职责 |
|-------|------|
| `@dev-op-metal` | Metal GPU kernel 开发指南（新增算子标准流程） |
| `@dev-heap-metal` | heap-plat 开发指南（张量生命周期） |
| `@dev-io-metal` | io-metal I/O 平面开发指南（print/save/load 操作） |
| `@dev-vm` | VM 核心开发指南（原生算子、CALL 翻译） |

## 开发 Skills

| Skill | 用途 |
|-------|------|
| `add-metal-kernel` | 新增 Metal GPU kernel 的 7 步引导式工作流 |
| `debug-vthread` | vthread 执行调试 (Redis 检查、PC 跟踪、常见问题) |
| `dual-opcode-audit` | VM ↔ op-plat opcode 一致性审计 |
| `debug-kvspace` | KV Space 联调状态检查 (redis-cli 查堆/栈) |

## 代码审计

- `@audit` — 全组件代码质量审计 agent，检查 10 条强制规则：
  0. 零 panic（VM 是常驻服务，任何 panic 都会导致崩溃）
  1. 严禁吞错误（所有 error 返回值必须检查，禁用 `_` 丢弃）
  2. 严禁裸 continue 吞错误（循环中错误必须至少 log）
  3. 外部协议一致性（同类型后端统一命名/协议）
  4. 错误必须可追溯（SetError 必须含 vtid/pc/msg 上下文）
  5. JSON 序列化/反序列化错误必须检查
  6. Redis 操作返回的 error 必须检查
  7. C++ 特定规则 (std::stoll 必须 try/catch, shm 操作必须检查返回值)
  8. Python 特定规则 (禁止裸 except)
  9. Go 特定规则 (禁止 panic, 禁止 _ 丢弃 error, if 显式求值)

## Git

> **上游 squash merge → origin 上 N 个 commit 变 1 个 → push 冲突。**
>
> **铁律：未提交修改 + 未 push 的 commit → 绝不丢弃。已 push 的 commit → 丢弃（上游已 squash）。**
>
> 解法：`git stash && git fetch origin && git rebase origin/main && git stash pop`
>
> 详见 `.claude/git.md`

