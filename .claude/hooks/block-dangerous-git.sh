#!/usr/bin/env bash
# PreToolUse hook: 拦截高风险 git 命令
# 触发条件: Bash 工具调用包含禁止命令时阻止执行

INPUT=$(cat)

# 提取命令文本
CMD=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null)

DANGEROUS_PATTERNS=(
  'git reset --hard'
  'git push --force'
  'git push -f'
  'git clean -fd'
  'git clean -fdx'
  'git checkout -- \.'
  'git restore \.'
  'rm -rf \.git'
  'git reset .*origin'
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$CMD" | grep -qE "$pattern"; then
    echo '{
      "decision": "block",
      "reason": "高危 git 命令已被 .claude/hooks 拦截。规则见 .claude/git.md"
    }'
    exit 2
  fi
done

# 放行
echo '{"decision": "allow"}'
