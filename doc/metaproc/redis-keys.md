# Redis Key 设计列表

> DeepX 元程系统使用的所有 Redis key 路径、value 类型和约定。
> Redis 作为 KV 空间，key 是路径空间的唯一标识，value 存储结构化数据 (JSON)。

## 1. 路径空间总览

```
/ (根)
├── src/func/          函数源码层 (pysdk 写入, 人类可读)
├── op/                算子注册与编译产物
│   ├── op-cuda/
│   ├── op-metal/
│   └── op-cpu/
├── vthread/           vthread 执行状态 (栈)
├── sys/               系统信息
├── cmd/               命令队列
├── notify/            通知队列
├── done/              完成通知队列
├── lock/              互斥锁
├── models/            堆变量 (示例)
├── data/              堆变量 (示例)
└── checkpoints/       堆变量 (示例)
```

---

## 2. 源码层: /src/func/

pysdk 写入的函数源码，dxlang 人类可读文本格式。

| Key | Value 类型 | 示例 | 说明 |
|-----|-----------|------|------|
| `/src/func/<name>` | string | `(add(A:tensor, B:tensor) -> (C:tensor))` | 函数签名 (dxlang) |
| `/src/func/<name>/N` | string | `add(A, B) -> ./C` | 第 N 条指令 |
| `/src/func/<name>/N/true/0` | string | `add(A, B) -> ./Y` | 分支 true 子块第 0 条 |
| `/src/func/<name>/N/false/0` | string | `sub(A, B) -> ./Y` | 分支 false 子块第 0 条 |
| `/src/func/<name>/N/body/0` | string | `add(./i, ./a) -> ./b` | for 循环体第 0 条 |

**指令格式 (左读右写):**
```
opcode(read_param_1, read_param_2, ...) -> write_param_1, write_param_2
```

**示例:**
```
/src/func/gemm                                    = "(gemm(A:tensor<f32>, B:tensor<f32>, alpha:f32, beta:f32, C:tensor<f32>) -> (Y:tensor<f32>))"
/src/func/gemm/0                                  = "matmul(A, B) -> ./Y"
/src/func/gemm/1                                  = "mul(./Y, alpha) -> ./Y"
/src/func/gemm/2                                  = "mul(C, beta) -> ./C"
/src/func/gemm/3                                  = "add(./Y, ./C) -> ./Y"
```

---

## 3. 编译层: /op/<backend>/func/

编译器读取 `/src/func/` 源码后产出的后端专属编译产物。VM CALL 时读取此层。

### 3.1 函数编译产物

| Key | Value 类型 | 示例 | 说明 |
|-----|-----------|------|------|
| `/op/<backend>/func/<name>` | string | `(gemm(...) -> (...))` | 编译后的函数签名 |
| `/op/<backend>/func/<name>/N` | string | `fused_matmul_add(A, B, b) -> ./out` | 编译后第 N 条指令 |

**注意:** 编译层的指令序号 N 和源码层的 N 可能不同（融合后 N 减少，拆分后 N 增加）。

```
示例 (CUDA 融合):
  /op/op-cuda/func/gemm/0 = "fused_matmul_mul_mul_add(A, B, alpha, C, beta) -> ./Y"
  (源码层 4 条 → 编译层 1 条)

示例 (Tensor 并行拆分):
  /op/op-cuda/func/forward/0 = "slice(A, 0, 512) -> ./A_shard0"
  /op/op-cuda/func/forward/1 = "slice(A, 512, 1024) -> ./A_shard1"
  /op/op-cuda/func/forward/2 = "matmul(./A_shard0, W) -> ./out0 @gpu0"
  /op/op-cuda/func/forward/3 = "matmul(./A_shard1, W) -> ./out1 @gpu1"
  /op/op-cuda/func/forward/4 = "concat(./out0, ./out1) -> ./out"
```

### 3.2 算子注册 (程序级 — 同一程序的所有进程实例共享)

| Key | Value 类型 | 示例 | 说明 |
|-----|-----------|------|------|
| `/op/<program>/list` | JSON array | `["matmul", "add", "relu", "sigmoid"]` | 该程序支持的全部算子 |
| `/op/<program>/<opname>` | JSON object | 见下方 | 单个算子的元数据 |

**算子元数据格式:**
```json
{
    "category": "matmul | elementwise | reduce | changeshape | activation | fused | init",
    "dtype": ["f32", "f16", "bf16"],
    "max_shape": [8192, 8192, 8192],
    "fusion_group": "linear"
}
```

**融合算子额外字段:**
```json
{
    "category": "fused",
    "dtype": ["f32", "f16"],
    "replaces": ["matmul", "add", "relu"]
}
```

**完整示例:**
```
/op/op-cuda/list                               = ["matmul", "add", "mul", "relu", "sigmoid", "fused_matmul_add_relu"]
/op/op-cuda/matmul                             = {"category":"matmul", "dtype":["f32","f16","bf16"], "max_shape":[8192,8192,8192], "fusion_group":"linear"}
/op/op-cuda/relu                               = {"category":"activation", "dtype":["f32","f16","bf16"]}
/op/op-cuda/fused_matmul_add_relu              = {"category":"fused", "dtype":["f32","f16"], "replaces":["matmul","add","relu"]}

/op/op-metal/list                              = ["add", "sub", "mul", "div", "relu", "sigmoid", "tanh", "zeros", "ones"]
/op/op-metal/add                               = {"category":"elementwise", "dtype":["f32","f16","i32"]}
```

---

## 4. 执行层: /vthread/

VM 管理的 vthread 执行状态。指令展开为二维坐标 `[addr0, addr1]`，命名槽位为平级子 key。

### 4.1 Vthread 自身

| Key | Value 类型 | 示例 | 说明 |
|-----|-----------|------|------|
| `/vthread/<vtid>` | JSON object | `{"pc":"[3,0]", "status":"running"}` | vtid 自身状态：pc + status + 可选 error |

**pc 字段格式:**
- 根栈: `"[0,0]"`, `"[3,0]"`
- 子栈: `"[n,0]/[0,0]"`, `"[n,0]/[3,0]"`
- 深层嵌套: `"[2,0]/[1,0]/[0,0]"`

**status 字段值:**

| status | 含义 |
|--------|------|
| `init` | 已创建，待 VM 拾取 |
| `running` | VM 正在调度执行 |
| `wait` | 等待异步操作 (op-plat / heap-plat 完成) |
| `error` | 执行出错 |
| `done` | 执行完毕，可 GC |

**error 字段 (仅 status=error 时存在):**
```json
{
    "pc": "[3,0]",
    "status": "error",
    "error": {
        "code": "GPU_OOM",
        "message": "out of memory: requested 2GB, available 1.5GB"
    }
}
```

### 4.2 指令坐标 (二维寻址)

| Key 模式 | Value 类型 | 含义 | addr1 规则 |
|----------|-----------|------|-----------|
| `/vthread/<vtid>/[addr0, 0]` | string | 操作码 | `0` = opcode |
| `/vthread/<vtid>/[addr0, -1]` | string | 读参数 #1 | `-N` = 第 N 个读取参数 |
| `/vthread/<vtid>/[addr0, -2]` | string | 读参数 #2 | |
| `/vthread/<vtid>/[addr0, 1]` | string | 写参数 #1 | `+N` = 第 N 个写入参数 |
| `/vthread/<vtid>/[addr0, 2]` | string | 写参数 #2 | |

addr0 是序列维整数，表示指令在栈帧内的顺序位置。

**示例 (指令 `add(./a, ./b) -> ./c`):**
```
/vthread/1/[0, 0]  = "add"
/vthread/1/[0,-1]  = "./a"
/vthread/1/[0,-2]  = "./b"
/vthread/1/[0, 1]  = "./c"
```

**示例 (指令 `matmul(A, B) -> ./Y`):**
```
/vthread/1/[0, 0]  = "matmul"
/vthread/1/[0,-1]  = "/models/A"
/vthread/1/[0,-2]  = "/models/B"
/vthread/1/[0, 1]  = "./Y"
```

**CALL 指令示例:**
```
/vthread/1/[0, 0]  = "call"
/vthread/1/[0,-1]  = "gemm"            # func_name
/vthread/1/[0,-2]  = "/models/A"       # 实参1
/vthread/1/[0,-3]  = "/models/B"       # 实参2
/vthread/1/[0,-4]  = "1.0"             # 实参3 (立即数)
/vthread/1/[0, 1]  = "./out"           # 返回值绑定的槽位
```

### 4.3 子栈 (CALL 产生)

```
/vthread/<vtid>/[n,0]/                       ← 子栈根
/vthread/<vtid>/[n,0]/[0,0]                  ← 子栈指令 #0 操作码
/vthread/<vtid>/[n,0]/[0,-1]                 ← 子栈指令 #0 读参数 #1
/vthread/<vtid>/[n,0]/[1,0]                  ← 子栈指令 #1 操作码
/vthread/<vtid>/[n,0]/[m,0]/[0,0]            ← 更深层嵌套

嵌套路径与 pc 的对应:
  根栈:           pc = "[0,0]"
  根栈 CALL n 后: pc = "[n,0]/[0,0]"
  子栈 CALL m 后: pc = "[n,0]/[m,0]/[0,0]"
```

### 4.4 命名槽位 (局部变量)

命名槽位是 `/vthread/<vtid>/` 下的平级子 key，与指令坐标 `[addr0, addr1]` 互不嵌套。

| Key | Value | 说明 |
|-----|-------|------|
| `/vthread/<vtid>/<name>` | 基础类型 或 tensor 元信息 | 局部变量，与 dxlang 源码变量名一致 |

**基础类型直存:**
```
/vthread/1/a       = 1            (int)
/vthread/1/b       = 3.14         (float)
/vthread/1/flag    = true         (bool)
/vthread/1/name    = "hello"      (string, < 1KB)
```

**Tensor 类型:**
```
/vthread/1/mm = {"dtype":"f32", "shape":[1024,256], "device":"gpu0", "address":{"type":"shm","shm_name":"/deepx_t_abc123"}, "byte_size":1048576}
```

命名槽位与指令坐标的平级关系:
```
/vthread/1/              ← vtid 自身
/vthread/1/[0,0]         ← 指令坐标
/vthread/1/[0,-1]        ← 指令坐标
/vthread/1/[0,1]         ← 指令坐标
/vthread/1/a             ← 命名槽位 (平级)
/vthread/1/mm            ← 命名槽位 (平级)
/vthread/1/[0,0]/[0,0]   ← 子栈 (嵌套)
```

---

## 5. 系统路径: /sys/

| Key | Value 类型 | 示例 | 说明 |
|-----|-----------|------|------|
| `/sys/vtid_counter` | int | `42` | vthread ID 自增计数器 (原子 INCR) |
| `/sys/config` | JSON object | `{"max_vthreads": 100, "timeout_ms": 30000}` | 全局配置 |
| `/sys/op-plat/<instance>` | JSON object | 见下方 | op-plat 进程实例注册 |
| `/sys/heap-plat/<instance>` | JSON object | 见下方 | heap-plat 进程实例注册 |
| `/sys/vm/<id>` | JSON object | `{"status":"running", "pid":12349, "started_at":1714000004}` | VM 实例注册 |

**进程级注册格式:**

```
/sys/op-plat/cuda:0    = {"program":"op-cuda", "device":"gpu0", "status":"running", "load":0.3, "pid":12345, "started_at":1714000000}
/sys/op-plat/cuda:1    = {"program":"op-cuda", "device":"gpu1", "status":"running", "load":0.7, "pid":12346, "started_at":1714000001}
/sys/op-plat/metal:0   = {"program":"op-metal", "device":"gpu0", "status":"running", "load":0.1, "pid":12347, "started_at":1714000002}
/sys/heap-plat/metal:0 = {"program":"heap-metal", "device":"gpu0", "status":"running", "pid":12348, "started_at":1714000003}
/sys/vm/0              = {"status":"running", "pid":12349, "started_at":1714000004}
```

**实例命名规则:** `<program>:<instance>`，如 `cuda:0`, `metal:0`，对应命令队列 `cmd:op-cuda:0`。

---

## 6. 命令队列: /cmd/ 和 /done/ 和 /notify/

使用 Redis List (FIFO 队列)。生产者 RPUSH，消费者 RPOP / BLPOP。

### 6.1 op-plat 命令队列

| Key | 消费方 | 生产者 | 说明 |
|-----|--------|--------|------|
| `cmd:op-cuda:0` | op-cuda 实例 0 | VM | CUDA 计算指令 |
| `cmd:op-metal:0` | op-metal 实例 0 | VM | Metal 计算指令 |
| `cmd:op-cpu:0` | op-cpu 实例 0 | VM | CPU 计算指令 |

**消息格式 (JSON):**
```json
{
    "vtid": "1",
    "pc": "[3,0]",
    "opcode": "matmul",
    "inputs": [
        {
            "key": "/vthread/1/a",
            "dtype": "f32",
            "shape": [1024, 512],
            "address": {
                "node": "n1",
                "device": "gpu0",
                "type": "shm",
                "shm_name": "/deepx_t_abc123",
                "byte_size": 2097152
            }
        }
    ],
    "outputs": [
        {
            "key": "/vthread/1/c",
            "dtype": "f32",
            "shape": [1024, 256],
            "address": {
                "node": "n1",
                "device": "gpu0",
                "type": "shm",
                "shm_name": "/deepx_t_def456",
                "byte_size": 1048576
            }
        }
    ],
    "params": {}
}
```

**批量消息 (可选):**
```json
{
    "batch": [
        {"pc": "[3,0]", "opcode": "add", "inputs": [...], "outputs": [...]},
        {"pc": "[4,0]", "opcode": "mul", "inputs": [...], "outputs": [...]}
    ]
}
```

### 6.2 heap-plat 命令队列

| Key | 消费方 | 说明 |
|-----|--------|------|
| `cmd:heap-metal:0` | heap-metal | Metal 内存管理指令 |
| `cmd:heap-cuda:0` | heap-cuda | CUDA 内存管理指令 |
| `cmd:heap-cpu:0` | heap-cpu | CPU 内存管理指令 |

**newtensor:**
```json
{"vtid":"1", "pc":"[0,0]", "op":"newtensor", "key":"/models/weights", "dtype":"f32", "shape":[1024,512], "device":"gpu0"}
```

**deltensor:**
```json
{"vtid":"1", "pc":"[5,0]", "op":"deltensor", "key":"/models/weights"}
```

**clonetensor:**
```json
{"vtid":"1", "pc":"[0,0]", "op":"clonetensor", "src":"/models/weights", "dst":"/models/weights_gpu1", "device":"gpu1"}
```

### 6.3 完成通知队列

| Key | 消费方 | 生产者 | 说明 |
|-----|--------|--------|------|
| `done:<vtid>` | VM | op-plat / heap-plat | vthread 的完成通知 |

**成功:**
```json
{"pc": "[3,0]", "status": "ok", "outputs_updated": [{"key": "/vthread/1/c", "new_shape": [1024, 256]}]}
```

**失败:**
```json
{"pc": "[3,0]", "status": "error", "error": {"code": "GPU_OOM", "message": "out of memory: requested 2GB, available 1.5GB"}}
```

### 6.4 VM 唤醒通知

| Key | 消费方 | 生产者 | 说明 |
|-----|--------|--------|------|
| `notify:vm` | VM | pysdk | 新 vthread 创建后唤醒 VM |

```json
{"event": "new_vthread", "vtid": "42"}
```

---

## 7. 堆变量 (隐式命名空间)

除保留路径外，KV 空间中的所有其他路径均为堆变量。value 存储 tensor 元信息。

### 7.1 Tensor 元信息格式

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
| dtype | string | 是 | `f32`, `f16`, `f64`, `bf16`, `i8`, `i16`, `i32`, `i64`, `u8`, `bool` |
| shape | array[int] | 是 | 如 `[1024, 512]` 或 `[2, 3, 224, 224]` |
| byte_size | int | 是 | element_count × dtype_size |
| device | string | 是 | `gpu0`, `gpu1`, `cpu` |
| address | object | 是 | 物理地址信息 |
| address.node | string | 是 | 机器标识 |
| address.type | string | 是 | `shm`, `gpu`, `cpu` |
| address.shm_name | string | 是 | POSIX shm 名称 |
| ctime | int | 否 | 创建时间 (unix timestamp) |
| version | int | 否 | 版本号 (每次写入递增) |

### 7.2 命名约定建议

```
/models/<model_name>/<layer>/<param_name>
/data/<dataset>/<split>
/checkpoints/<run_id>/<step>
/rl/<env_name>/<buffer_name>
```

示例:
```
/models/bert/encoder_0/weights  = {"dtype":"f32", "shape":[768,3072], ...}
/models/bert/encoder_0/bias     = {"dtype":"f32", "shape":[3072], ...}
/data/cifar10/train             = {"dtype":"f32", "shape":[50000,3,32,32], ...}
```

---

## 8. 锁

| Key | Value | 操作 | 说明 |
|-----|-------|------|------|
| `/lock/<resource>` | holder 标识符 | `SET NX EX` / Lua DEL | 排他锁 |

锁 key 命名约定: `/lock/<resource>`

```
/lock/tensor:/models/weights      → 保护 weights 的并发修改
/lock/vthread:1                   → 保护 vthread 1 的并发拾取 (也可用 WATCH/MULTI/EXEC)
```

---

## 9. Key 操作复杂度与约束

### 9.1 批量操作

| 模式 | Redis 命令 | 说明 |
|------|-----------|------|
| 读多个指令 | `MGET key1 key2 ...` | 一次往返 |
| 读多个 tensor 元信息 | `MGET /vthread/1/a /vthread/1/b /models/W` | 一次往返 |
| 写多条指令 | `Pipeline: SET ... SET ...` | 一次往返 |
| 原子抢占 vthread | `WATCH ... MULTI ... EXEC` | 事务 |
| 扫描 vthread | `KEYS /vthread/*` | 避免高频调用 |
| 递归删除子栈 | `KEYS prefix*` → 批量 `DEL` | 两步 |

### 9.2 值大小约束

| 类型 | 约束 | 建议 |
|------|------|------|
| 短字符串 | < 1KB | 指令、签名 |
| JSON 对象 | < 10KB | tensor 元信息、状态 |
| Tensor 实际数据 | **不存 Redis** | 存外部 shm，Redis 仅存引用 |

### 9.3 命名约束

- key 中不使用空格，用 `/` 分隔层级
- 指令坐标: `[addr0, addr1]`，addr0 非负整数，addr1 整数
- 命名槽位: 字母数字 + 下划线，与 dxlang 变量名一致
- 实例名: `<program>:<instance>`，如 `cuda:0`, `metal:0`

---

## 10. Key 路径速查表

```
# 源码层
/src/func/<name>                         string    函数签名
/src/func/<name>/N                       string    第 N 条指令
/src/func/<name>/N/true/0                string    分支 true 子块
/src/func/<name>/N/false/0               string    分支 false 子块
/src/func/<name>/N/body/0                string    for 循环体

# 编译层
/op/<backend>/func/<name>                string    编译后签名
/op/<backend>/func/<name>/N              string    编译后指令

# 算子注册 (程序级，所有实例共享)
/op/<program>/list                       array     算子列表
/op/<program>/<opname>                   object    算子元数据

# 执行层
/vthread/<vtid>                          object    {pc, status, error?}
/vthread/<vtid>/[addr0, 0]               string    操作码
/vthread/<vtid>/[addr0, -N]              string    读参数 (N=1,2,...)
/vthread/<vtid>/[addr0, +N]              string    写参数 (N=1,2,...)
/vthread/<vtid>/[addr0,0]/[sub0,0]       string    子栈操作码
/vthread/<vtid>/<name>                   any       命名槽位 (局部变量)

# 系统
/sys/vtid_counter                        int       vthread ID 计数器
/sys/config                              object    全局配置
/sys/op-plat/<instance>                  object    op-plat 进程注册
/sys/heap-plat/<instance>                object    heap-plat 进程注册
/sys/vm/<id>                             object    VM 实例注册

# 命令队列 (List)
cmd:op-<backend>:<instance>              list      op-plat 命令队列
cmd:heap-<backend>:<instance>         list      heap-plat 命令队列
done:<vtid>                              list      完成通知队列
notify:vm                                list      VM 唤醒通知

# 锁
/lock/<resource>                         string    排他锁

# 堆变量 (隐式)
/<any_non_reserved_path>                 object    tensor 元信息
```
