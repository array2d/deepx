# 项目智能体

开发指引见 `.claude/CLAUDE.md` 的文档表。

## 可用 agent

| Agent | 文件 | 职责 |
|-------|------|------|
| `audit` | `audit-vm.md` | 审计 deepx **全组件**代码质量（VM/op-plat/heap-plat/pysdk），覆盖 Go/C++/Python，检查 10 条强制规则 |
| `dev-op-metal` | `dev-op-metal.md` | op-metal Metal GPU 算子开发专家。指导新增 kernel、dtype 覆盖、dispatch 流程 |
| `dev-heap-metal` | `dev-heap-metal.md` | heap-plat 张量生命周期开发专家。指导 newtensor/deltensor/clonetensor、引用计数、shm 管理 |
| `dev-io-metal` | `dev-io-metal.md` | io-metal I/O 平面开发专家。指导 print/save/load 等 tensor I/O 操作开发 |
| `dev-vm` | `dev-vm.md` | VM 核心开发专家。指导原生算子新增、CALL 翻译、状态机、并发安全 |

使用方式：对话中输入 `@dev-op-metal 新增 gelu 算子` 即可触发对应开发流程指引。
