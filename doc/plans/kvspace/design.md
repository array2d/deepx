# KVSpace — deepx 统一键值寻址空间

> 以 Unix 目录规则为灵感，为 deepx 运行时建立统一的分层命名空间。
> 核心概念：**虚程 (Virtual Process, VP)** — 一种不以 OS 进程为边界的计算实体，
> 其元数据与代码存于 KVSpace，堆数据存于 GPU/CPU 显存，由 VM 调度器分发执行。

## 1. 什么是虚程

传统 OS 进程 = 内核管理的内存空间 + 线程 + 文件描述符。它的边界是单机。

虚程 = KVSpace 命名空间中的一个子树 + 分布于各机器的实际资源。

```
┌─ KVSpace (Redis) ────────────────────────────────────────────┐
│                                                                │
│  /vproc/42/                    ← 虚程 42 的"灵魂"             │
│    method/                     ← 代码区 (dxlang 算子序列)      │
│    heap/                       ← 堆区 (tensor 元数据)          │
│    stack/                      ← 栈区 (临时 tensor 元数据)     │
│    code/                       ← dxlang 字节码/IR              │
│    config/                     ← 虚程配置                      │
│    status                      ← 运行状态                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │  VM 调度器读取 code/，分发到 executor
         ▼
┌─ GPU 显存 / CPU 内存 (实际机器) ──────────────────────────────┐
│                                                                │
│  /deepx_t_xxx (POSIX shm)      ← heap/ 中 tensor 的实际数据    │
│  MTLBuffer / cudaMalloc ptr    ← stack/ 中临时 tensor 的数据   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

| 对比 | OS 进程 | 虚程 |
|------|--------|------|
| 身份 | PID (内核分配) | VPID (KVSpace 分配) |
| 内存 | 虚拟地址空间 (内核管理) | 元数据在 Redis，数据在 GPU shm |
| 代码 | ELF 加载到内存 | dxlang IR 存于 KVSpace |
| 调度 | 内核调度器 | VM 调度器 → executor |
| 生命周期 | fork/exec/exit | create/run/stop (KVSpace 操作) |
| 边界 | 单机 | 跨机 (元数据可达，数据在显存) |

## 2. 虚程的命名空间

仿 Unix `/proc/`，虚程挂在 `/vproc/` 下：

```
/vproc/                           所有虚程
├── /vproc/list                   [1, 42, 99]
│
├── /vproc/<vpid>/                虚程 <vpid>
│   │
│   ├── info                      虚程元信息
│   │   { "vpid": 42, "name": "train_mnist", "type": "training",
│   │     "host": "n1", "ctime": 1714000000 }
│   │
│   ├── status                    运行时状态
│   │   { "state": "running", "uptime_sec": 3600,
│   │     "ops_executed": 12345, "errors": 0 }
│   │
│   ├── code/                     ★ 代码区 (dxlang 算子序列)
│   │   ├── code/main             { "lang": "dxlang", "ir": "..." }
│   │   ├── code/init             { "ops": ["new_weights", "new_bias"] }
│   │   └── code/forward          { "ops": ["matmul", "add", "relu"] }
│   │
│   ├── method/                   ★ 方法区 (注册的算子能力)
│   │   ├── method/list           ["matmul", "relu", "softmax"]
│   │   └── method/<name>         { "author": "miaobyte", "dtype": ["f32"],
│   │                                "category": "elementwise" }
│   │
│   ├── heap/                     ★ 堆区 (长生命周期 tensor 元数据)
│   │   ├── heap/list             ["weights", "bias", "embedding"]
│   │   └── heap/<name>           堆 tensor 描述
│   │       { "dtype": "f32", "shape": [1024,512], "byte_size": 2097152,
│   │         "device": "gpu0", "shm": "/deepx_t_a1b2c3",
│   │         "refcount": 1, "ctime": 1714000000 }
│   │
│   ├── stack/                    ★ 栈区 (临时 tensor，用完即回收)
│   │   └── stack/<name>          { "dtype": "f32", "shape": [1024],
│   │                                "device": "gpu0", "shm": "..." }
│   │
│   ├── config/                   虚程级配置
│   │   ├── config/default_device  "gpu0"
│   │   └── config/log_level       "debug"
│   │
│   ├── metrics/                  虚程级指标
│   │   ├── metrics/ops_per_sec    123.4
│   │   └── metrics/gpu_util_pct   87.5
│   │
│   └── log/                      虚程日志
│       └── log/last               ["[INFO] forward pass completed", ...]
│
└── /vproc/self/                  当前上下文的虚程 → /vproc/42
```

## 3. 虚程 vs executor 进程

虚程和 executor 进程是两个正交概念：

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  虚程 A (训练)           虚程 B (推理)          虚程 C (RL)      │
│  /vproc/1/              /vproc/2/              /vproc/3/         │
│  code/forward: [m,m,r]  code/predict: [m,r,s]  code/rollout: []  │
│  heap/: [w,b]           heap/: [w]             heap/: [state]    │
│      │                       │                      │             │
│      │  VM 调度器读取 code，分配给 executor                     │
│      ▼                       ▼                      ▼             │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              executor 进程池                          │        │
│  │  op-metal (PID 100)  op-metal (PID 101)  op-cuda     │        │
│  │  执行 matmul,add     执行 relu,softmax    执行 ...    │        │
│  │  GPU0                GPU0                 GPU1        │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
│  虚程 ≠ executor 进程。一个虚程的代码可被多个 executor 执行。   │
│  executor 是"干活的"，虚程是"谁要干什么"的元信息。              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**关键关系：**
- 一个虚程的 code 区存储**需要做什么**（dxlang 算子序列）
- VM 调度器读取 code，将算子分发给有能力的 executor
- heap tensor 的元数据在虚程名下，实际数据在 GPU shm
- executor 进程自己也以虚程形式注册在 `/vproc/` 下（作为"服务虚程"）

## 4. VM 调度器

```
┌─ VM 调度器 ─────────────────────────────────────────────────────┐
│                                                                   │
│  WATCH /vproc/+/code/  ← 监听所有虚程的 code 区变化              │
│                                                                   │
│  当虚程 42 SET code/forward = [matmul, add, relu]:               │
│    1. 解析算子序列                                                │
│    2. GET /vproc/42/heap/weights → 找到 tensor 位置               │
│    3. GET /op/matmul/impl → 找到能执行 matmul 的 executor 虚程    │
│    4. GET /op/add/impl                                            │
│    5. 将算子 + tensor 引用打包为任务                              │
│    6. 发送到 executor 的 UDP 端口 (来自 executor 虚程的 info)     │
│    7. 监听 executor 返回，更新 /vproc/42/status                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 5. 完整命名空间树（全局视角）

```
/                                  KVSpace 根
│
├── /vproc/                        ★ 所有虚程
│   ├── /vproc/list               [1, 42, 99, 100, 101]
│   ├── /vproc/<vpid>/            每个虚程的完整子树 (见 §2)
│   └── /vproc/self/              当前上下文的虚程
│
├── /dev/                          设备树
│   ├── /dev/list                  ["cpu", "gpu0", "gpu1"]
│   ├── /dev/cpu/info / mem/*
│   └── /dev/gpu<N>/info / mem/* / temp
│
├── /tensor/                       全局堆 tensor 注册表
│   ├── /tensor/list               ["model/weights", "rl/state"]
│   └── /tensor/<name>/
│       ├── meta, data, refcount
│       ├── owner → "/vproc/42"    所属虚程
│       ├── shm                    实际存储位置 (shm name)
│       └── lock                   跨虚程互斥锁
│
├── /op/                           全局算子注册表
│   ├── /op/list
│   └── /op/<name>/
│       ├── signature              输入输出签名
│       └── impl                   ["/vproc/100/method/matmul", ...]
│                                  (哪些虚程实现了该算子)
│
├── /session/                      训练/推理会话
│   └── /session/<id>/
│       ├── state, graph, params, metrics
│       └── vproc → "/vproc/42"    关联的虚程
│
├── /sys/                          系统信息
│   └── version, uptime, hostname, mem/*
│
└── /config/                       全局配置
    └── default_dtype, default_device, log_level, redis_addr
```

## 6. Value 类型

| 路径模式 | Value 类型 | 示例 |
|----------|-----------|------|
| `*/list` | JSON 字符串数组 | `["matmul","add"]` |
| `*/info` `*/meta` `*/status` | JSON 对象 | `{"dtype":"f32"}` |
| `*/code/*` | JSON 对象 (dxlang IR) | `{"lang":"dxlang","ops":[...]}` |
| `*/data` | 二进制 bytes | tensor 原始数据 |
| `*/config/*` | 标量 | `"debug"` `42` |
| `*/metrics/*` | 数值 | `123.4` |
| `*/lock` `*/refcount` | 整数 | `3` |
| 中间目录 | LIST 返回子节点名 | — |

## 7. 操作原语

```
GET    <path>           → value
SET    <path> <value>   → ok/error
LIST   <path>           → [child_names]
WATCH  <path>           → event_stream     (值变化时推送)
DELETE <path>           → ok/error
LINK   <src> <dst>      → ok/error         (符号链接, 如 /vproc/self)
LOCK   <path> <vpid>    → ok/error
UNLOCK <path> <vpid>    → ok/error
```

## 8. 虚程生命周期

```
CREATE /vproc/42/info = {name:"train_mnist", type:"training"}
  → VM 分配 VPID，初始化子树

SET /vproc/42/code/init = {ops:["new_weights","new_bias"]}
SET /vproc/42/code/forward = {ops:["matmul","add","relu"]}
  → 代码区就绪

SET /vproc/42/config/default_device = "gpu0"
  → 配置就绪

SET /vproc/42/status = {state:"ready"}
  → 虚程就绪，等待调度

VM 调度器 WATCH 到 state=ready → 开始执行 code/init
  → 分配 heap tensor → SET /vproc/42/heap/weights = {...}

VM 调度器执行 code/forward:
  → GET /op/matmul/impl → 找到 executor
  → 发送任务到 executor UDP
  → executor 通过 /tensor/weights/shm 获取数据，执行 GPU 计算
  → 结果写回，更新 /vproc/42/metrics/

DELETE /vproc/42
  → refcount=0 的 heap tensor 触发 GC
  → shm_unlink, 清理子树
```

## 9. 虚程间通信：通过 /tensor 共享数据

```
虚程 A (rollout)                      虚程 B (training)
/vproc/1/                             /vproc/2/
  │                                     │
  │ SET /tensor/rl_state/meta = {...}   │
  │ SET /tensor/rl_state/shm = "..."    │
  │ SET /tensor/rl_state/refcount = 2   │
  │                                     │
  │ ──── WATCH 通知 ──────────────────→ │ GET /tensor/rl_state/shm
  │                                     │ shm_open → 读取 GPU 结果
  │                                     │ 训练 → 更新权重
  │ ←─── /tensor/weights 更新 ←──────── │
  │                                     │
```

## 10. 实现分层

| 层 | 存储 | 内容 |
|----|------|------|
| KVSpace (Redis) | Redis Hash/String | 所有元数据：虚程 info、tensor meta、code IR、config |
| heapmem-* | POSIX shm + GPU | tensor 实际数据，KVSpace 中仅存 shm name |
| executor | 进程内存 | 执行状态，通过 /vproc/<vpid>/status 对外暴露 |
| VM 调度器 | 独立进程 | 读取 code/，WATCH 状态变化，分发任务 |

## 11. 设计哲学

1. **虚程是 deepx 的一等公民**。所有计算以虚程为组织单元，不直接操作 executor 进程。

2. **代码即数据**。dxlang IR 作为虚程 code/ 区的值，可被 VM 调度器读取、缓存、优化。

3. **堆与栈分离**。堆 tensor (heap/) 跨虚程共享，生命周期由引用计数管理；栈 tensor (stack/) 在单次计算中临时存在。

4. **WATCH 驱动**。VM 调度器不轮询，而是 WATCH 虚程状态变化，事件驱动执行流。

5. **Unix 路径语义**。LIST、GET、SET、LINK 构成完整的管理接口。监控面板只需 LIST /vproc/ 即可发现所有虚程。
