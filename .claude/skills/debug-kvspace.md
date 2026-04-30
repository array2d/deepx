# skill: debug-kvspace → KV Space 联调状态检查

用 `redis-cli` 在联调时快速检查 deepx 元程的堆 (tensor 元信息 / heap-plat) 和栈 (vthread 执行状态)。

## 前置条件

- Redis 运行中，默认地址 `127.0.0.1:16379` (由 `REDIS_ADDR` 环境变量控制)
- `redis-cli` 已安装 (本项目环境: `/opt/homebrew/bin/redis-cli` 8.6.2)

## 速连

```bash
# Claude 可直接执行 redis-cli，无需 sandbox 放行
redis-cli -p 16379 PING
# → PONG
```

## 栈检查 (vthread — VM 管理的元程执行栈)

vthread key 树结构见 `doc/metaproc/redis-keys.md` §4。

### 概览

```bash
# 列出全部 vthread
redis-cli -p 16379 KEYS "/vthread/*" | grep -v '\[.*\]'

# 看某个 vthread 的完整状态 (pc + status + error)
redis-cli -p 16379 GET /vthread/<vtid> | jq .

# 看 vthread 全部子 key (指令坐标 + 命名槽位 + 子栈)
redis-cli -p 16379 KEYS "/vthread/<vtid>/*" | sort
```

### PC & 指令

```bash
# 当前执行到哪条指令
redis-cli -p 16379 GET /vthread/<vtid> | jq '.pc'

# 根栈第 N 条指令的操作码 + 读写参数
redis-cli -p 16379 MGET \
  /vthread/<vtid>/[N,0] \
  /vthread/<vtid>/[N,-1] \
  /vthread/<vtid>/[N,-2] \
  /vthread/<vtid>/[N,1]

# 扫描子栈 (CALL 后多层嵌套)
redis-cli -p 16379 KEYS "/vthread/<vtid>/\[*" | sort
```

### 命名槽位 (局部变量)

```bash
# 看所有非坐标子 key (局部变量 / tensor 引用)
redis-cli -p 16379 KEYS "/vthread/<vtid>/*" | grep -v '\[.*\]'

# 看某个槽位的值 (基础类型直存, tensor 存 JSON 元信息)
redis-cli -p 16379 GET /vthread/<vtid>/<name> | jq .
```

### 状态解读

| status | 含义 | 排查方向 |
|--------|------|---------|
| `init` | 已创建，待 VM 拾取 | 检查 VM 是否运行 |
| `running` | 正在执行 | 正常 |
| `wait` | 等待异步操作 | 检查 `cmd:exop-metal:*` / `cmd:heap-metal:*` 队列 |
| `error` | 执行出错 | 查看 `error` 字段详情 |
| `done` | 执行完毕 | 正常，可 GC |

## 堆检查 (tensor 元信息 & heap-plat)

deepx 堆由 **heap-plat 进程** 管理，tensor 元信息存 Redis，实际数据存 POSIX shm。

### 堆变量 (tensor 元信息)

```bash
# 列出所有堆变量 (排除保留路径)
redis-cli -p 16379 KEYS "*" | grep -v -E '^(/src|/vthread|/sys|/cmd:|/done:|/notify:|/lock:|/op/)' | head -50

# 看某个 tensor 的完整元信息
redis-cli -p 16379 GET /models/<name> | jq .
# → {"dtype":"f32","shape":[1024,512],"byte_size":2097152,"device":"gpu0","address":{"type":"shm","shm_name":"/deepx_t_abc123"}}

# 批量看关键字段
redis-cli -p 16379 MGET /models/A /models/B | while read line; do echo "$line" | jq -c '{dtype,shape,device}'; done
```

### heap-plat 进程状态

```bash
# 查看 heap-plat 实例注册
redis-cli -p 16379 GET /sys/heap-plat/metal:0 | jq .

# 列出所有 heap 实例
redis-cli -p 16379 KEYS "/sys/heap-plat/*"
```

### heap 命令队列

```bash
# 队列长度 (堆积 >0 说明 heap-plat 未响应)
redis-cli -p 16379 LLEN cmd:heap-metal:0
	redis-cli -p 16379 LLEN cmd:io-metal:0

# 查看全部待处理命令
redis-cli -p 16379 LRANGE cmd:heap-metal:0 0 -1

# 查看一条命令详情 (newtensor / deltensor / clonetensor)
redis-cli -p 16379 LINDEX cmd:heap-metal:0 0 | jq .
```

### io-plat 进程状态

```bash
# 查看 io-plat 实例注册
redis-cli -p 16379 GET /sys/io-plat/io-metal:0 | jq .

# io 命令队列
redis-cli -p 16379 LLEN cmd:io-metal:0
redis-cli -p 16379 LRANGE cmd:io-metal:0 0 -1
```

### shm 存在性验证

```bash
# 从 Redis 拿到 shm_name 后验证 shm 是否真实存在
ls -la /tmp/deepx_t_* 2>/dev/null
```

## 联调联合检查 (堆 + 栈一站式)

联调时最常见的快速检查路径：

```bash
# 1. 确认全部平台进程在线
redis-cli -p 16379 KEYS "/sys/*" | sort

# 2. 确认 vthread 状态
redis-cli -p 16379 GET /vthread/1 | jq '{pc,status}'

# 3. 若 wait → 查命令队列堆积
redis-cli -p 16379 LLEN cmd:exop-metal:0
redis-cli -p 16379 LLEN cmd:heap-metal:0
	redis-cli -p 16379 LLEN cmd:io-metal:0

# 4. 若 error → 查看错误详情
redis-cli -p 16379 GET /vthread/1 | jq '.error'

# 5. 查看堆 tensor 是否完整
redis-cli -p 16379 KEYS "/models/*" | while read k; do
  echo -n "$k → "; redis-cli -p 16379 GET "$k" | jq -c '{dtype,shape,device}'
done

# 6. 检查完成通知
redis-cli -p 16379 KEYS "done:*"
redis-cli -p 16379 LRANGE done:1 0 -1
```

## 快速重置

```bash
make reset-redis
# 等价于 redis-cli -p 16379 FLUSHDB
```

## 参考

- Redis key 完整规范: `doc/metaproc/redis-keys.md`
- vthread 调试工作流: `.claude/skills/debug-vthread.md`
- heap-plat 开发: `doc/heap-plat/`
