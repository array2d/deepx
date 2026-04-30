# skill: dual-opcode-audit → 双端 opcode 一致性审计

检查 VM 分派的 opcode 与 op-plat 注册的算子列表是否匹配。

## 审计目标

确保三层之间 opcode 命名一致:

1. **VM 分派层**: `ir` 包中识别的 opcode (IsComputeOp / IsLifecycleOp / IsControlOp / IsNativeOp)
2. **op-plat 注册层**: `/op/exop-metal/list` + `/op/op-cuda/list` 中注册的算子
3. **测试层**: `testdata/` 中 `.dx` 文件使用的 opcode

## 审计步骤

### Step 1: 收集 VM 端 opcode

列出所有 VM 会分发到 op-plat 的 opcode (不在 native/lifecycle/control 中的):

```bash
# 从 VM 测试文件统计实际使用的 opcode
grep -rhoP '^\s*\w+\(' executor/vm/testdata/ | sort -u | sed 's/($//'
```

### Step 2: 收集 op-plat 端 opcode

```bash
# 从 Redis 读取 (需 Redis 运行 + op-plat 已注册)
redis-cli LRANGE "/op/exop-metal/list" 0 -1
redis-cli LRANGE "/op/op-cuda/list" 0 -1
```

或从源码静态收集:
```bash
# exop-metal main.cpp 中 RPUSH 的算子
grep -A1 'RPUSH.*list' executor/exop-metal/src/client/main.cpp | grep '"' | grep -oP '"\K[^"]+' | grep -v '/op/' | grep -v 'RPUSH'
```

### Step 3: 交叉比对

| opcode | VM 使用 | exop-metal 注册 | op-cuda 注册 | CPU fallback | 状态 |
|--------|---------|---------------|--------------|--------------|------|
| add    | ✅      | ✅            | ?            | —            | ✅   |
| newop  | ✅      | ❌            | ❌            | —            | ❌   |

### Step 4: 识别问题

**问题类型 A**: VM 分派但无 op-plat 注册 → `route.Select()` 返回 "no op-plat supports"
**问题类型 B**: op-plat 注册但 VM 无测试覆盖 → 死代码
**问题类型 C**: opcode 拼写不一致 (e.g., "equalscalar" vs "eq_scalar")
**问题类型 D**: VM native 与 op-plat 都处理同一 opcode → 优先级冲突 (VM native 优先)

### Step 5: 自动检查脚本

```bash
#!/bin/bash
# 快速 opcode 一致性检查
echo "=== VM native ops ==="
grep ':"' executor/vm/internal/ir/native.go | grep -oP '"\K[^"]+' | sort > /tmp/vm_native.txt

echo "=== VM test ops ==="
grep -rhoP '^\s*\w+\(' executor/vm/testdata/ | sed 's/($//' | sort -u > /tmp/vm_test.txt

echo "=== exop-metal registered ==="
# (需要从 Redis 或 main.cpp 中提取)
grep -A1 'RPUSH.*list' executor/exop-metal/src/client/main.cpp | \
  grep '"[a-z]' | grep -oP '"\K[^"]+' | grep -v 'list' | sort > /tmp/op_metal.txt

echo "=== Diff: VM test vs exop-metal ==="
comm -23 /tmp/vm_test.txt /tmp/op_metal.txt
echo "(VM test ops NOT in exop-metal)"

comm -13 /tmp/vm_test.txt /tmp/op_metal.txt
echo "(exop-metal ops NOT in VM tests)"
```

## 合规标准

每个 compute opcode 必须满足:
- 至少一个 op-plat 后端注册 (metal 或 cuda 或 cpu)
- VM 有对应的 dispatch 路径 (Compute 函数)
- 测试覆盖 (testdata 中有 .dx 文件)
