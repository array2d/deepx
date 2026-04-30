# 项目钩子

deepx 事件拦截与自动化。

## 可用钩子

| 钩子 | 触发事件 | 动作 |
|------|---------|------|
| `post-build-op-metal` | `/build-op-metal` 完成后 | 验证 default.metallib 存在 + 运行 shm 测试 |
| `post-build-vm` | `/build-vm` 完成后 | go vet 检查 + 单元测试结果确认 |

## 实现方式

钩子逻辑嵌入在对应命令脚本中（如 `test-op-metal.sh` 同时做构建+测试）。

如需扩展为独立钩子文件，在此目录下创建 `on-<event>.sh` 并在 settings 中注册。
