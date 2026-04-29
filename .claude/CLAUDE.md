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
- `/build-op-cuda` — 构建 CUDA 计算平面
- `/build-op-metal` — 构建 Metal 计算平面
- `/build-heap-metal` — 构建 Metal 堆管理平面
- `/reset-redis` — 重置 Redis 测试环境 (FLUSHDB)

## 代码审计
- `@audit-vm` — VM 代码质量审计 agent，检查 6 条强制规则：
  0. 零 panic（VM 是常驻服务，任何 panic 都会导致崩溃）
  1. 严禁吞错误（所有 error 返回值必须检查，禁用 `_` 丢弃）
  2. 严禁裸 continue 吞错误（循环中错误必须至少 log）
  3. IF 条件显式求值（不允许隐式 truthiness）
  4. 错误必须可追溯（SetError 必须含 vtid/pc/msg 上下文）
  5. JSON 序列化/反序列化错误必须检查
  6. Redis 操作返回的 error 必须检查
