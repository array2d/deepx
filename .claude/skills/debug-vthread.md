# skill: debug-vthread → vthread 执行调试

排查 vthread 执行卡住、报错、或结果不符合预期的问题。

## 前置条件

- Redis 运行中（含测试数据）
- VM binary 已构建 (`/build-vm`)

## 调试工作流

### 1. 快速诊断

用 `redis-cli` 检查 vthread 状态:

```bash
# 查看 vthread 状态
redis-cli GET /vthread/<vtid>

# 查看当前 PC 指令
redis-cli MGET /vthread/<vtid>/[0,0] /vthread/<vtid>/[0,-1] /vthread/<vtid>/[0,1]

# 查看所有执行层 key
redis-cli KEYS "/vthread/<vtid>/*"
```

**状态解读**:
| Status | 含义 | 排查方向 |
|--------|------|---------|
| `init` | VM 未拾取 | 检查 VM worker 是否运行; 检查 picker 日志 |
| `running` | 正在执行 | 正常 |
| `wait` | 等待 op/head-plat | 检查 `cmd:exop-metal:0` / `cmd:heap-metal:0` 队列 |
| `error` | 出错 | GET `/vthread/<vtid>` 查看 error 详情 |
| `done` | 完成 | 正常 |

### 2. PC 跟踪

查看当前执行到哪条指令:
```bash
# 方式 1: 直接读状态
redis-cli GET /vthread/<vtid> | jq .

# 方式 2: 扫描子栈 (CALL 路径)
redis-cli KEYS "/vthread/<vtid>/*" | sort
```

PC 格式: `[0,0]` (根栈) → `[2,0]/[0,0]` (子栈) → `[2,0]/[3,0]/[0,0]` (深层子栈)

### 3. 查看编译层 vs 执行层

对比编译层指令和翻译后的执行层:
```bash
# 编译层 (源码)
redis-cli GET "/src/func/<funcname>" 
redis-cli KEYS "/src/func/<funcname>/*" | sort -t/ -k5 -n | xargs redis-cli GET

# 执行层 (翻译后)
redis-cli KEYS "/vthread/<vtid>/<pc>/*" | while read k; do echo "$k: $(redis-cli GET "$k")"; done
```

### 4. 异步任务追踪

op-plat / heap-plat 的异步通信:
```bash
# 查看 op-plat 命令队列长度
redis-cli LLEN "cmd:exop-metal:0"

# 查看 done 队列
redis-cli KEYS "done:*"

# 查看算子系统注册
redis-cli LRANGE "/op/exop-metal/list" 0 -1
redis-cli GET "/sys/op-plat/exop-metal:0"
```

### 5. 单步调试

使用 VM single-run 模式逐 vthread 执行:
```bash
./executor/vm/build/vm run <vtid> [redis_addr]
```

输出包含最终 PC、Status、Error 信息。

### 6. 常见问题模式

| 症状 | 常见原因 | 解决 |
|------|---------|------|
| vthread 永远 `wait` | op-plat / heap-plat 未启动 | 启动对应进程 |
| CALL 失败 "func not found" | 编译层或源码层缺少函数定义 | SET `/src/func/<name>` 或 `/op/<backend>/func/<name>` |
| "route: no op-plat supports" | 算子未在任何 op-plat 注册 | 检查 `/op/*/list` |
| "BLPOP timeout" | op-plat 响应超时 (30s) | 增大 timeout 或优化 kernel |
| PC 全部为 `[0,0]` | pysdk 写入指令时未使用递增编号 | 检查 `/src/func/<name>/N` 的 N 是否连续递增 |

### 7. 重置测试环境

```bash
/reset-redis
```
清空所有 vthread / src / op / sys / done / cmd key 后重新测试。

## 参考

- dxlang 引号约定: `doc/dxlang/README.md` — 字符串 `"..."` / Key `'...'` / 变量 无引号
- Redis key 规范: `doc/metaproc/redis-keys.md`
