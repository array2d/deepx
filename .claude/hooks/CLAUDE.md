# 项目钩子

deepx 事件拦截与自动化。

## 已注册钩子

| 钩子 | 类型 | 触发条件 | 动作 |
|------|------|---------|------|
| `block-dangerous-git.sh` | `PreToolUse(Bash)` | 任何 Bash 调用包含高风险 git 命令时 | **拦截执行**，返回错误阻止命令运行 |

### 拦截的高风险命令

- `git reset --hard`
- `git push --force` / `git push -f`
- `git clean -fd` / `git clean -fdx`
- `git checkout -- .`
- `git restore .`
- `rm -rf .git`

拦截规则详见 `.claude/git.md`。

## 添加新钩子

在此目录下创建脚本文件（如 `on-post-build.sh`），然后在 `.claude/settings.local.json` 中注册：

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{"type": "command", "command": "bash .claude/hooks/your-hook.sh"}] }
    ]
  }
}
```

支持的 hook 事件: `PreToolUse`, `PostToolUse`, `Notification`, `Stop`
