# deepx

基于元程 (Metaproc) 的分布式深度学习运行时。以 Redis KV 空间为全局状态存储，5 核协作：

| 核心 | 角色 | 目录 |
|------|------|------|
| Redis | KV 空间 | — |
| pysdk | 算法前端，注册源码到 `/src/func/` | `front/py/` |
| op-plat | 被动消费指令，执行 GPU/CPU 张量计算 | `executor/op-cuda/` `executor/op-metal/` |
| heap-plat | 管理 tensor 生命周期 (shm 分配/释放) | `executor/heap-metal/` |
| VM | CALL 翻译、指令路由、状态推进、原生算子求值 | `executor/vm/` (Go) |

## 文档

| 目录 | 用途 | 文件 |
|------|------|------|
| `doc/metaproc/` | 整体架构、Redis key、开发指南 | `spec-v1.md` `deepx-design.md` `redis-keys.md` `dev-heap-plat.md` `dev-op-plat.md` `dev-pysdk.md` `dev-vm.md` |
| `doc/vm/` | VM 设计、dxlang 语法、调度器 | `README.md` `scheduler.md` `dxlang/README.md` |
| `doc/heap-plat/` | 堆管理平面 (tensor 生命周期) | `README.md` `heap-metal.md` `heap-cuda.md` `heap-cpu.md` |
| `doc/op-plat/` | 计算平面 (算子注册、GPU kernel) | `README.md` `op-metal.md` `op-cuda.md` `op-cpu.md` |

按任务查阅: 架构→`metaproc/spec-v1.md` · op开发→`op-plat/` · heap开发→`heap-plat/` · VM开发→`vm/` · pysdk→`metaproc/dev-pysdk.md` · heap-plat开发→`metaproc/dev-heap-plat.md` · op-plat开发→`metaproc/dev-op-plat.md` · VM开发→`metaproc/dev-vm.md`

## 术语

见 `.claude/glossary.md`（元程核心术语速查）。

## 构建 / 测试

### 构建

| 命令 | 说明 |
|------|------|
| `/build-op-metal` | 构建 Metal 计算平面 (cmake + Metal shader 编译) |
| `/build-op-cuda` | 构建 CUDA 计算平面 |
| `/build-heap-metal` | 构建 Metal 堆管理平面 |
| `/build-vm` | 构建 VM + loader (Go compile + 单元测试) |

### 测试

| 命令 | 说明 |
|------|------|
| `/test-vm` | 运行 VM 单元测试 |
| `/test-integration` | 运行 VM 集成测试 (需要 Redis，纯 VM 算子) |
| `/test-real` | 运行真实联调测试 (Redis + heap-metal + op-metal 子进程) |
| `/test-op-metal` | 构建 op-metal 并运行 shm 跨进程测试 |
| `/pipeline` | 完整联调流水线: build → start → reset → test-real → stop |
| `/reset-redis` | 重置 Redis 测试环境 (FLUSHDB) |

### 联调测试架构

真实联调测试详见 `executor/vm/integration_real_test.go`，使用 Go 的 `os/exec` 管理子进程生命周期:

- `TestIntegrationReal_Lifecycle` — heap-metal: newtensor / deltensor
- `TestIntegrationReal_Compute`  — heap-metal + op-metal: create → add → delete
- `TestIntegrationReal_IO`       — op-metal: print / save / load via SHM
- `TestIntegrationReal_CrossCall`— VM 原生函数组合 (diamond)
- `TestIntegrationReal_E2E`      — 端到端: 序列调用 + GPU/native 混合

所有 `integration_real_test.go` 通过 `executor/Makefile` 管理:
```bash
cd executor
make pipeline REDIS_ADDR=127.0.0.1:16379  # 一键完整流程
make test-real REDIS_ADDR=127.0.0.1:16379 # 仅运行测试 (需先 start-services)
make start-services                        # 后台启动 plats
make stop-services                         # 停止
make status                                # 查看状态
```

### 加载示例到 KV Space

```bash
# 构建后 loader 位于 /tmp/deepx-vm/loader

# 加载单个文件
./tmp/deepx-vm/loader load example/dxlang/lifecycle/full.dx

# 加载整个目录
./tmp/deepx-vm/loader load example/dxlang/nn/

# 列出已注册函数
./tmp/deepx-vm/loader ls

# 加载并执行 (需要 VM + heap-plat + op-plat 在运行)
./tmp/deepx-vm/loader run example/dxlang/native/arith/add.dx native_arith "./a:2" "./b:3" -- "./c"
```

## 开发 Agents

| Agent | 职责 |
|-------|------|
| `@dev-op-metal` | Metal GPU kernel 开发指南（新增算子标准流程） |
| `@dev-heap-metal` | heap-plat 开发指南（张量生命周期） |
| `@dev-vm` | VM 核心开发指南（原生算子、CALL 翻译） |

## 开发 Skills

| Skill | 用途 |
|-------|------|
| `add-metal-kernel` | 新增 Metal GPU kernel 的 7 步引导式工作流 |
| `debug-vthread` | vthread 执行调试 (Redis 检查、PC 跟踪、常见问题) |
| `dual-opcode-audit` | VM ↔ op-plat opcode 一致性审计 |

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

