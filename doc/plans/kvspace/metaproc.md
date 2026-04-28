# 元程 (Metaproc) 定义与实现设计

> 元程是 deepx 的一等计算实体。其元数据 (meta) 存于 KVSpace，
> 其实际数据 (tensor) 存于 GPU/CPU 显存，其代码 (dxlang) 亦存于 KVSpace，
> 由 VM 调度器分发到 executor 进程执行。

## 1. 定义

```
元程 = 身份 (VPID) + 命名空间 (/vproc/<vpid>/) + 四个区 + 生命周期

四个区:
  code/    代码区 — dxlang 算子序列，定义"要做什么"
  method/  方法区 — 该元程注册的算子能力（如果它同时是 executor）
  heap/    堆区   — 长生命周期 tensor 元数据，跨元程共享
  stack/   栈区   — 临时 tensor 元数据，单次计算内有效
```

### 1.1 本质：元程是什么

元程不是 OS 进程。OS 进程有独立虚拟地址空间，由内核调度。

元程是一个 **KVSpace 中的 JSON 文档集合**，描述了一段计算任务的完整状态。
"运行"一个元程，就是把它的 code/ 翻译为对 executor 的调度请求。

```
OS 进程 ≈ 一段在 CPU 上跑的机器码 + 内核管理的内存

元程   ≈ KVSpace 里的一棵子树 + GPU 上的 tensor 数据
         + VM 调度器根据子树里的 code/ 去指挥 executor
```

### 1.2 元程的类型

| 类型 | 用途 | code/ 有吗 | heap/ 有吗 | 示例 |
|------|------|-----------|-----------|------|
| 计算元程 | 执行训练/推理任务 | ✅ | ✅ | train_mnist, infer_bert |
| 服务元程 | 注册算子能力，等待调度 | ❌ | ❌ | op-metal, op-cuda |
| 数据元程 | 管理数据加载/预处理 | ✅ | ✅ | dataloader_cifar10 |
| 会话元程 | 管理训练状态，协调多个计算元程 | ✅ | ❌ | rl_coordinator |

服务元程是 executor 进程在 KVSpace 中的"代言人"。它不存 code，但 method/ 列出了该 executor 能执行的算子。VM 调度器通过 method/ 发现能力。

## 2. 元程的四区详解

### 2.1 code/ — 代码区

存 dxlang 算子序列。格式：

```
/vproc/<vpid>/code/
├── init        { "lang": "dxlang", "seq": [
│                   {"op": "new", "name": "weights", "dtype": "f32", "shape": [1024,512]},
│                   {"op": "new", "name": "bias",    "dtype": "f32", "shape": [512]}
│               ]}
│
├── forward     { "lang": "dxlang", "seq": [
│                   {"op": "matmul", "A": "input", "B": "weights", "C": "hidden"},
│                   {"op": "add",    "A": "hidden", "B": "bias",    "C": "hidden"},
│                   {"op": "relu",   "A": "hidden", "C": "output"}
│               ]}
│
├── backward    { "lang": "dxlang", "seq": [
│                   {"op": "relu_grad",   "A": "hidden", "grad": "grad_output", "C": "grad_hidden"},
│                   {"op": "matmul_grad", "A": "input", "grad": "grad_hidden", "C": "grad_weights"},
│                   {"op": "reduce_sum",  "A": "grad_hidden", "C": "grad_bias"}
│               ]}
│
└── checkpoint  { "lang": "dxlang", "seq": [
                    {"op": "save", "name": "weights", "path": "/checkpoints/step_1000.pt"},
                    {"op": "save", "name": "bias",    "path": "/checkpoints/step_1000.pt"}
                ]}
```

**语义：**
- code/ 下的每个 key 是一个**命名代码块**（init, forward, backward, checkpoint...）
- 每个代码块是 dxlang 算子序列
- VM 调度器一次执行一个代码块
- 代码块之间通过 `status.next_code` 串行

### 2.2 method/ — 方法区

仅**服务元程**有意义。列出该 executor 能计算的算子及签名：

```
/vproc/100/method/
├── list       ["matmul", "add", "relu", "softmax"]
├── matmul     { "author": "miaobyte", "category": "matmul",
│                "dtype": ["f32", "f16"], "device": "gpu0",
│                "capability": { "max_shape": [8192,8192,8192] }}
├── add        { "author": "miaobyte", "category": "elementwise",
│                "dtype": ["f32", "f16", "i32"], "device": "gpu0" }
└── relu       { "author": "cublas", "category": "activation",
                 "dtype": ["f32", "f16"], "device": "gpu0" }
```

VM 调度器把 `/op/matmul/impl` 指向所有注册了 matmul 的元程，运行时根据负载、设备亲和性选择。

### 2.3 heap/ — 堆区

长生命周期 tensor 的元数据。数据本身在 GPU/CPU，KVSpace 仅存**描述**：

```
/vproc/42/heap/
├── list       ["weights", "bias", "embedding"]
├── weights    { "dtype": "f32", "shape": [1024, 512],
│                "byte_size": 2097152, "device": "gpu0",
│                "shm": "/deepx_t_a1b2c3",        ← 数据所在
│                "refcount": 2,                    ← 引用计数
│                "ctime": 1714000000,
│                "state": "ready" }
├── bias       { "dtype": "f32", "shape": [512],
│                "byte_size": 2048, "device": "gpu0",
│                "shm": "/deepx_t_d4e5f6",
│                "refcount": 1 }
└── embedding  { "dtype": "f32", "shape": [10000, 256],
                 "byte_size": 10240000, "device": "gpu0",
                 "shm": "/deepx_t_g7h8i9",
                 "refcount": 1 }
```

**关键设计：**
- `shm` 字段是 POSIX 共享内存名，指向实际数据
- heap/ 与 /tensor/ 的关系：`/tensor/<name>` 是全局注册表，heap/ 是元程视角
- GET `/tensor/weights/shm` 等价于 GET `/vproc/42/heap/weights` → `.shm`
- refcount 管理生命周期。refcount=0 时 heapmem 回收 shm

**heap 的分配：**
```
元程 code/init 中有 {"op":"new","name":"weights","dtype":"f32","shape":[1024,512]}

VM 调度器接到 new 指令:
  1. 如果 /tensor/weights 已存在 → refcount++
  2. 否则 → 通知 heapmem: shm_open, 分配 GPU buffer
  3. SET /vproc/42/heap/weights = {shm:"...", refcount:1, ...}
  4. SET /tensor/weights/meta = {...}
```

### 2.4 stack/ — 栈区

临时 tensor，作用域为单次代码块执行。执行完即释放：

```
/vproc/42/stack/
├── hidden     { "dtype": "f32", "shape": [1024, 512],
│                "device": "gpu0", "shm": "/deepx_t_tmp_001" }
├── output     { "dtype": "f32", "shape": [1024, 512],
│                "device": "gpu0", "shm": "/deepx_t_tmp_002" }
└── grad_hidden { ... }
```

**语义：**
- stack tensor 在每个代码块开始前分配，执行后释放
- 不在 /tensor/ 全局注册（不跨元程共享）
- shm 用临时名称，不持久化
- 如果某 stack tensor 需要升级为堆 tensor（如 checkpoint 保存），显式 `op: promote`

## 3. 元程与 /tensor/ 全局注册表的关系

```
/tensor/ 是全局的、跨元程的 tensor 目录
/vproc/<vpid>/heap/ 是该元程"持有"或"引用"的 tensor 视角

发起方:
  /vproc/1/heap/weights  ← SET 时同步到 → /tensor/weights/

使用方:
  /vproc/2/ 想用 weights:
    GET /tensor/weights/meta → {shm, dtype, shape}
    GET /tensor/weights/refcount → 3  (自动 inc)
    shm_open(meta.shm) → 拿到 GPU buffer

    ... 计算完毕 ...

    SET /tensor/weights/refcount = 2  (自动 dec)
```

**所有权模型：**
- owner: 创建该 tensor 的元程。owner 退出时通知 refcount--。
- user: 引用该 tensor 的元程。user 退出时 refcount--。
- heapmem: 当 refcount=0 且 state="deleted"，回收 shm。

## 4. 元程状态机

```
                    ┌─────────┐
                    │  init   │  ← CREATE 完成，子树已建立
                    └────┬────┘
                         │ code/init 执行完成
                    ┌────▼────┐
               ┌───►│  ready  │◄──────────┐
               │    └────┬────┘           │
               │         │ VM 调度        │
               │    ┌────▼────┐           │
               │    │ running │           │
               │    └────┬────┘           │
               │     ┌───┴───┐            │
               │     │       │            │
           code/next  │   error          │
               │ ┌────▼──┐ ┌─▼──────┐    │
               │ │  wait │ │ error  │    │
               │ └───┬───┘ └───┬────┘    │
               │     │         │         │
               │  WATCH 触发   │ retry   │
               │     │         │         │
               └─────┴─────────┘─────────┘
                         │
                    ┌────▼────┐
                    │ stopped │  ← DELETE /vproc/<vpid>
                    └─────────┘
```

状态说明：

| 状态 | 含义 | 允许的操作 |
|------|------|-----------|
| init | 正在执行 code/init，分配 heap tensor | 内部操作 |
| ready | 就绪，等待 VM 调度 | SET code/* |
| running | 正在执行某个代码块 | 只读 status |
| wait | 等待外部条件（如 /tensor/X/lock 释放） | WATCH |
| error | 执行出错 | 读取 error 信息，retry |
| stopped | 已终止，子树待 GC | DELETE |

**状态存储：**
```
/vproc/<vpid>/status = {
    "state": "running",
    "current_code": "forward",
    "current_op_idx": 3,
    "next_code": "backward",
    "started_at": 1714000000,
    "error": ""
}
```

## 5. VM 调度器

### 5.1 架构

```
┌────────────────── VM 调度器 ───────────────────────────────┐
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │ WATCH 引擎   │   │ 算子路由器   │   │ executor 池管理    │  │
│  │             │   │             │   │                  │  │
│  │ 监听:       │   │ matmul → ?  │   │ op-metal#0: idle │  │
│  │ /vproc/+/   │──▶│ add    → ?  │──▶│ op-metal#1: busy │  │
│  │ status      │   │ relu   → ?  │   │ op-cuda#0:  idle │  │
│  └─────────────┘   └─────────────┘   └──────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 调度流程

```
1. WATCH /vproc/+/status
   检测到 state="ready", current_code="forward"
   
2. GET /vproc/42/code/forward
   解析 seq: [matmul, add, relu]
   
3. 对 seq 中每个 op:
   a. GET /op/<op>/impl → ["/vproc/100/method/<op>", "/vproc/101/method/<op>"]
   b. 选择负载最低的 executor: /vproc/100 (op-metal#0)
   
4. 构造任务包:
   {
     "vpid": 42,
     "op": "matmul",
     "inputs":  {"A": "/vproc/42/stack/input", "B": "/vproc/42/heap/weights"},
     "outputs": {"C": "/vproc/42/stack/hidden"},
     "dtype": "f32"
   }
   
5. 发送到 executor 的 UDP 端口 (来自 /vproc/100/info.udp_port)

6. executor 执行:
   - 根据 inputs 中的路径 GET shm name
   - shm_open → MTLBuffer/CUDA ptr
   - GPU kernel dispatch
   - 结果写入 outputs shm
   - UDP 回复完成

7. VM 收到回复 → SET /vproc/42/status.current_op_idx += 1
   继续 seq 中的下一个 op
   
8. seq 全部完成 → SET /vproc/42/status.state = "ready"
                  SET /vproc/42/status.next_code = "backward"
```

### 5.3 算子到 executor 的映射

```
/op/matmul/impl = ["/vproc/100/method/matmul", "/vproc/201/method/matmul"]

VM 路由逻辑:
  GET /vproc/100/status.executor_load → 0.8 (busy)
  GET /vproc/201/status.executor_load → 0.2 (idle)
  → 选 /vproc/201
  GET /vproc/201/info.udp_port → 9091
  → 发送到 9091
```

## 6. 元程间通信

### 6.1 通过 /tensor 共享数据

```
元程 A (rollout_env):             元程 B (train_model):
  │                                 │
  │ 执行 code/step → 产出 state     │
  │ SET /tensor/rl_state/meta       │
  │ SET /tensor/rl_state/shm        │
  │ SET /tensor/rl_state/refcount=2 │
  │                                 │
  │ ─── WATCH /tensor/rl_state ──→  │ 收到通知
  │                                 │ code/train_step 被触发
  │                                 │ GET /tensor/rl_state/shm
  │                                 │ GPU 训练...
  │                                 │ 更新 /vproc/B/heap/weights
  │                                 │
  │ ←── /tensor/weights 更新 ────── │ SET /tensor/weights/meta
  │ WATCH /tensor/weights           │
  │ GET 新权重 → 下一轮 rollout     │
```

### 6.2 同步原语

```
LOCK   /tensor/X/lock <vpid>   → 获取排他锁 (Redis SETNX)
UNLOCK /tensor/X/lock <vpid>   → 释放锁
WATCH  /tensor/X/lock          → 监听锁释放

元程 B 等待元程 A 完成:
  WATCH /vproc/A/status
  当 state="ready" → 开始自己的 code/forward
```

## 7. 代码区与执行模型

### 7.1 dxlang 序列结构

```
code/forward = {
  "lang": "dxlang",
  "seq": [
    {"op": "matmul",  "A": "$input",   "B": "heap:weights", "C": "stack:hidden"},
    {"op": "add",     "A": "stack:hidden", "B": "heap:bias",   "C": "stack:hidden2"},
    {"op": "relu",    "A": "stack:hidden2",                    "C": "$output"}
  ]
}
```

**前缀约定：**
- `$name` — 外部传入的形参 (代码块被调用时绑定)
- `heap:name` — 该元程 heap/ 下的 tensor
- `stack:name` — 该元程 stack/ 下的临时 tensor
- 无前缀 — 先查 stack，再查 heap

### 7.2 分支与条件

```
code/step = {
  "lang": "dxlang",
  "seq": [
    {"op": "forward"},
    {"if": {"condition": "loss < 0.01"},
     "then": [{"op": "save_checkpoint"}],
     "else": [{"op": "backward"}, {"op": "update"}]}
  ]
}
```

### 7.3 调用其他元程的代码块

```
code/train_epoch = {
  "lang": "dxlang",
  "seq": [
    {"call": {"vproc": "dataloader_1", "code": "next_batch",
              "bind": {"$batch": "stack:batch"}}},
    {"op": "forward", ...},
    {"op": "backward", ...}
  ]
}
```

## 8. 实现：KVSpace 存储方案

### 8.1 Redis 数据结构

```
元程子树映射到 Redis:

/vproc/<vpid>/info          → Redis Hash   HSET /vproc/42/info name "train" type "training"
/vproc/<vpid>/status        → Redis Hash   HSET /vproc/42/status state "running" ...
/vproc/<vpid>/code/<name>   → Redis String (JSON)  SET /vproc/42/code/forward '{...}'
/vproc/<vpid>/method/<name> → Redis Hash
/vproc/<vpid>/heap/<name>   → Redis Hash
/vproc/<vpid>/stack/<name>  → Redis Hash (临时，带 TTL)
/vproc/<vpid>/config/*      → Redis Hash
/vproc/<vpid>/metrics/*     → Redis Hash
/vproc/<vpid>/log/last      → Redis List (LPUSH, LTRIM 保留最近 N 条)

/vproc/list                 → Redis Set    SADD /vproc/list 42
/vproc/self                 → Redis String (符号链接: "/vproc/42")
```

### 8.2 目录操作映射

```
LIST /vproc/42/heap/   → Redis:  HKEYS /vproc/42/heap/list
                                 或 SMEMBERS /vproc/42/heap:names

LIST /vproc/           → Redis:  SMEMBERS /vproc/list

GET  /vproc/42/info    → Redis:  HGETALL /vproc/42/info → JSON

SET  /vproc/42/status  → Redis:  HSET /vproc/42/status state "running" ...

WATCH /vproc/42/status → Redis:  SUBSCRIBE /vproc/42/status:channel
                                 每次 HSET 时 PUBLISH
```

### 8.3 临时数据的 TTL

```
stack tensor:   EXPIRE 300  (5 分钟未访问则自动清理)
log entries:    LTRIM 保留最近 100 条
```

## 9. 与 heapmem-metal / op-metal 的集成点

```
┌─ heapmem-metal ────────────────────────────────────────────────┐
│                                                                 │
│  收到 Redis 指令: {"op":"create","name":"weights",...}          │
│  或其 WATCH /tensor/+/lock 被触发                               │
│                                                                 │
│  → shm_open + mmap + GPU buffer alloc                         │
│  → SET /tensor/weights/shm = "/deepx_t_xxx"                    │
│  → SET /vproc/42/heap/weights = {shm:"...", ...}               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─ op-metal ─────────────────────────────────────────────────────┐
│                                                                 │
│  收到 UDP 任务: {"vpid":42, "op":"matmul", ...}                │
│                                                                 │
│  解析 inputs:                                                   │
│    "A": "stack:input"  → GET /vproc/42/stack/input → shm        │
│    "B": "heap:weights" → GET /vproc/42/heap/weights → shm       │
│                                                                 │
│  → shm_open → newBufferWithBytesNoCopy → GPU dispatch           │
│  → 结果写 outputs stack tensor → 回复 VM                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 10. 创建元程的 API

```python
# Python 前端
kv = KVSpace()

# 创建计算元程
vpid = kv.create_metaproc(
    name="train_mnist",
    type="training",
    config={"default_device": "gpu0", "log_level": "debug"}
)
# → 返回 vpid=42, 自动建立 /vproc/42/ 子树

# 设置代码区
kv.set_code(vpid, "init", [
    {"op": "new", "name": "weights", "dtype": "f32", "shape": [784, 256]},
    {"op": "new", "name": "bias",    "dtype": "f32", "shape": [256]}
])

kv.set_code(vpid, "forward", [
    {"op": "matmul", "A": "$input", "B": "heap:weights", "C": "stack:hidden"},
    {"op": "add",    "A": "stack:hidden", "B": "heap:bias", "C": "stack:hidden2"},
    {"op": "relu",   "A": "stack:hidden2", "C": "$output"}
])

# 启动 → VM 调度器接管
kv.run_metaproc(vpid, code="init")
# init 执行完自动进入 forward

# 查看状态
status = kv.get(f"/vproc/{vpid}/status")
print(status["state"])  # → "running"

# 停止
kv.stop_metaproc(vpid)
```

## 11. 总结

| 概念 | 定义 | 存于 |
|------|------|------|
| 元程 | 计算实体的元信息集合 | KVSpace (Redis) |
| code/ | 要执行的 dxlang 算子序列 | KVSpace |
| method/ | executor 能算的算子列表 | KVSpace |
| heap/ | 长生命周期 tensor 描述 | KVSpace |
| stack/ | 临时 tensor 描述 (带 TTL) | KVSpace |
| tensor 数据 | 实际的 float 数组 | GPU/CPU shm |
| VM 调度器 | 读取 code, 路由到 executor | 独立进程 |
| executor | 执行 GPU kernel 的进程 | op-metal / op-cuda |
