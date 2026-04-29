# 项目智能体

开发指引见 `.claude/CLAUDE.md` 的文档表。

## 可用 agent

| Agent | 文件 | 职责 |
|-------|------|------|
| `audit-vm` | `audit-vm.md` | 审计 deepx **全组件**代码质量（VM/op-plat/heap-plat/pysdk），覆盖 Go/C++/Python，检查 10 条强制规则 |

使用方式：对话中输入 `@audit-vm 审计` 即可触发全项目代码审计。
