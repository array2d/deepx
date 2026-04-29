# heap-plat 设计

> heap-plat 是 DeepX 元程的 **堆管理平面**，负责 tensor 对象的生命周期管理。
> 本文档定义 heap-plat 的抽象契约和所有实现的共用规范。

## 1. 定位

在元程 5 核架构中，heap-plat 负责：

| 核心 | 角色 |
|------|------|
| KV 空间 (Redis) | 存储 tensor 元信息（dtype, shape, 物理地址） |
| heap-plat | 管理 tensor 外部存储（创建/删除/克隆 shm） |
| op-plat | 通过元信息中的物理地址访问 tensor 数据 |
| VM | 将 newtensor/deltensor 指令路由到 heap-plat |

```
VM PUSH newtensor ──→ cmd:heap-<backend>:<instance>
                         │
                    heap-plat 消费
                         │
                    分配 POSIX shm
                    分配 GPU/CPU buffer
                    写入 tensor 元信息到 Redis
                         │
VM BLPOP done:<vtid> ←── LPUSH 完成通知
```

## 2. 抽象契约

任何 heap-plat 实现（Metal/CUDA/CPU）必须满足以下契约。

### 2.1 必须实现的命令

| 命令 | 语义 | 输入 | 输出 |
|------|------|------|------|
| `newtensor` | 创建 tensor | key, dtype, shape, device | 分配存储，写入 Redis 元信息 |
| `deltensor` | 删除 tensor | key | 释放存储，删除 Redis key |
| `clonetensor` | 克隆 tensor 到指定设备 | src_key, dst_key, device | 分配+拷贝，写入新 Redis key |

### 2.2 通信模型

```
消费: RPOP / BLPOP cmd:heap-<backend>:<instance>
      ↓
执行: POSIX shm_open / shm_unlink + mmap
      ↓
写入: SET <key> = tensor 元信息 (JSON)
      ↓
通知: LPUSH done:<vtid> {pc, status, ...}
```

### 2.3 命令格式

**newtensor:**
```json
{
    "vtid": "1",
    "pc": "[0,0]",
    "op": "newtensor",
    "key": "/models/weights",
    "dtype": "f32",
    "shape": [1024, 512],
    "device": "gpu0"
}
```

**deltensor:**
```json
{
    "vtid": "1",
    "pc": "[5,0]",
    "op": "deltensor",
    "key": "/models/weights"
}
```

**clonetensor:**
```json
{
    "vtid": "1",
    "pc": "[0,0]",
    "op": "clonetensor",
    "src": "/models/weights",
    "dst": "/models/weights_gpu1",
    "device": "gpu1"
}
```

### 2.4 完成通知格式

```json
{
    "pc": "[0,0]",
    "status": "ok"
}
```

错误:
```json
{
    "pc": "[0,0]",
    "status": "error",
    "error": {
        "code": "SHM_ALLOC_FAILED",
        "message": "failed to allocate 2GB shared memory"
    }
}
```

### 2.5 进程注册

启动时向 `/sys/heap-plat/` 注册：

```
/sys/heap-plat/metal:0 = {"program":"heap-metal", "device":"gpu0", "status":"running", "pid":<pid>, "started_at":<ts>}
```

命名规则: `<program>:<instance>`，实例编号从 0 开始。

### 2.6 命令队列

VM 通过指定队列向特定实例发送指令：

```
cmd:heap-metal:0     → heap-metal 实例 0 (gpu0)
cmd:heap-metal:1     → heap-metal 实例 1 (gpu1)
cmd:heap-cuda:0      → heap-cuda 实例 0 (gpu0)
cmd:heap-cpu:0       → heap-cpu 实例 0 (cpu)
```

### 2.7 消费者循环 (所有实现共用)

```
1. 启动 → 注册 /sys/heap-plat/<instance>
2. 循环:
   a) BLPOP cmd:heap-<backend>:<instance> (超时 5s)
   b) 解析 JSON 命令
   c) 根据 op 字段分发:
      "newtensor" → handle_newtensor(req)
      "deltensor" → handle_deltensor(req)
      "clonetensor" → handle_clonetensor(req)
   d) LPUSH done:<vtid> 完成通知
3. 退出 → DELETE /sys/heap-plat/<instance>
```

## 3. Tensor 元信息格式 (统一)

所有 heap-plat 实现写入 Redis 的 tensor 元信息格式必须一致：

```json
{
    "dtype": "f32",
    "shape": [1024, 512],
    "byte_size": 2097152,
    "device": "gpu0",
    "address": {
        "node": "n1",
        "type": "shm",
        "shm_name": "/deepx_t_abc123"
    },
    "ctime": 1714000000,
    "version": 5
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| dtype | string | 是 | 见数据类型表 |
| shape | array[int] | 是 | 多维形状 |
| byte_size | int | 是 | 总字节数 = element_count × dtype_size |
| device | string | 是 | gpu0, cpu |
| address.type | string | 是 | 统一为 "shm" |
| address.shm_name | string | 是 | POSIX shm 路径，如 /deepx_t_<hex> |
| address.node | string | 是 | 机器标识 |
| ctime | int | 否 | 创建时间 |
| version | int | 否 | 每次更新递增 |

**数据类型大小:**

| dtype | bytes |
|-------|-------|
| f16, bf16 | 2 |
| f32, i32 | 4 |
| f64, i64 | 8 |
| i8, u8 | 1 |
| i16 | 2 |
| bool | 1 |

## 4. 各平台实现概览

| 实现 | 目录 | 状态 | 说明 |
|------|------|------|------|
| [heap-metal](heap-metal.md) | executor/heap-metal/ | 待开发 | macOS Metal 统一内存 |
| [heap-cuda](heap-cuda.md) | executor/heap-cuda/ | 待开发 | Linux CUDA GPU 显存 |
| [heap-cpu](heap-cpu.md) | executor/heap-cpu/ | 待开发 | 纯 CPU 内存 |

## 5. 待确定问题

| 问题 | 状态 |
|------|------|
| 引用计数 (refcount) | 暂不实现，由上层管理 |
| 跨节点 tensor 迁移 | 暂不实现 |
| 零拷贝 GPU 间传输 | 待评估 |
| shm_name 命名冲突 | 使用随机 hex 后缀 + EXISTS 检查 |
