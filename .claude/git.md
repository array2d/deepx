# Git

## 关键事实

- **origin**: `git@github.com:miaobyte/deepx.git`（GitHub 私人仓库）
- **上游合并策略**: squash merge → PR 的所有 commit 被压缩成一个，原始 commit SHA 全部丢弃

## push 冲突根因

origin 从上游同步了 squash 后的新历史（N 个 commit → 1 个 commit），与本地 commit 树分叉。

## 核心铁律

| 类型 | 处置 |
|------|------|
| 本地**未提交**的修改 | **绝不丢弃** |
| 本地**未 push** 的 commit | **绝不丢弃** |
| 本地**已 push** 的 N 个 commit | **丢弃**（上游已 squash 为 1 个） |

## 冲突解决

```bash
# 1. 保存未提交修改
git stash

# 2. 拉取 origin
git fetch origin

# 3. rebase：已 push 的 commit 会被跳过或可跳过；未 push 的 commit 被保留
git rebase origin/main
#    冲突时：已 push 的 → git rebase --skip
#            未 push 的 → 正常解决后 git rebase --continue

# 4. 恢复未提交修改
git stash pop
```

> **如果 rebase 冲突太多不想处理**：`git reset --hard origin/main` 丢弃全部本地 commit，然后手动拣回未 push 的 commit。
