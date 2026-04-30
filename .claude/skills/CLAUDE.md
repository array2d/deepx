# 项目技能

deepx 专用技能模块。提供引导式工作流，覆盖常见开发与调试任务。

## 可用技能

| 技能 | 文件 | 用途 |
|------|------|------|
| `add-metal-kernel` | `add-metal-kernel.md` | 新增 Metal GPU kernel 的 7 步引导式工作流 (shader→host→dispatch→注册→构建→测试) |
| `debug-vthread` | `debug-vthread.md` | vthread 执行调试指南 (Redis key 检查、PC 跟踪、异步追踪、单步运行、常见问题) |
| `dual-opcode-audit` | `dual-opcode-audit.md` | VM ↔ op-plat opcode 一致性审计 (交叉比对、问题分类、自动检查脚本) |

使用方式：对话中输入对应 skill 名称即可触发引导式工作流。例如："帮我新增一个 gelu Metal kernel，用 add-metal-kernel"。
