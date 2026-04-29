# 元程 — 数据结构篇

> 分布式计算系统中，数据结构即架构。
> 元程的 KV 路径空间就是它的"地址总线"，
> 路径约定就是它的"ABI"。本文档定义元程的全部数据结构及其关系。

---

## 1. 核心命题

```
程序 = 数据结构 + 函数 + 数据
       ↑            ↑       ↑
    元程的骨架    执行逻辑   流动内容
```

在传统 OS 中，数据结构是编译器实现细节：栈帧、堆段、代码段由 OS + 编译器联合管理，
程序员不可见。但在分布式系统中，**数据结构必须被显式定义并共享**——因为进程分布在
不同机器上，它们只能通过约定的数据结构来理解和协作。

元程的做法：把所有数据结构放在一个 **KV 空间** 里，用**路径**作为全局地址。
数据结构定义 = 路径空间的划分规则 + 每条路径上 value 的类型约定。

---

## 2. KV 空间 — 元程的地址空间

KV 空间是元程的**第一数据结构**。它等同于 OS 进程的虚拟地址空间，但有两个关键区别：

| | OS 虚拟地址空间 | 元程 KV 空间 |
|---|---|---|
| 寻址方式 | 64位线性地址 | 字符串路径（`/` 分层） |
| 可见性 | 进程私有 | 集群全局共享 |
| 读取粒度 | 字节 | key-value 对 |
| 组织方式 | 段（.text / .data / 栈） | 路径树（`/src/` `/vthread/` `/models/`） |
| 并发模型 | 线程 + 锁 | WATCH/MULTI/EXEC 事务 + 锁 |

```
/ (根)
├── src/func/         函数源码层
├── op/               算子注册与编译产物
│   ├── op-cuda/
│   ├── op-metal/
│   └── op-cpu/
├── vthread/          vthread 执行状态（栈）
├── sys/              系统信息
├── cmd/              命令队列
├── notify/           通知队列
├── done/             完成通知队列
├── lock/             互斥锁
├── models/           堆变量（示例）
├── data/             堆变量（示例）
└── checkpoints/      堆变量（示例）
```

KV 空间本身不规定路径语义——它是"白纸"。路径语义由**保留路径约定**赋予。

---

## 3. 保留路径 — 元程的"段"

如同 OS 的虚拟地址空间有 .text / .data / 栈 / 堆等固定区域，
元程的 KV 空间也有**保留路径**，每一段有明确的读写角色：

### 3.1 源码层 `/src/func/`

| 角色 | 内容 | 写入者 | 读取者 |
|---|---|---|---|
| 函数签名 | `(gemm(A, B) -> (Y))` dxlang 文本 | pysdk | VM, 编译器 |
| 指令序列 | `matmul(A, B) -> ./Y` dxlang 文本 | pysdk | VM, 编译器 |
| 控制流嵌套 | `/N/true/0`, `/N/body/0` 等 | pysdk | VM |

```
/src/func/<name>              → 函数签名
/src/func/<name>/0            → 第 0 条指令
/src/func/<name>/1            → 第 1 条指令
/src/func/<name>/N/true/0     → if 分支 true 子块
/src/func/<name>/N/body/0     → for 循环体
```

**属性**: 人类可读、平台无关。这是函数的"源文件"。

### 3.2 编译层 `/op/<backend>/func/`

| 角色 | 内容 | 写入者 | 读取者 |
|---|---|---|---|
| 编译后签名 | 同源码层或优化后签名 | 编译器 | VM |
| 编译后指令 | 融合/拆分/设备标注后的指令 | 编译器 | VM |

```
/op/op-cuda/func/<name>       → 编译后签名
/op/op-cuda/func/<name>/0     → 编译后第 0 条指令
/op/op-metal/func/<name>/0    → Metal 编译产物（可能不同）
```

**属性**: 后端专属。同一函数在不同设备上可能有不同的编译产物。

### 3.3 执行层 `/vthread/`

| 角色 | 内容 | 写入者 | 读取者 |
|---|---|---|---|
| vthread 状态 | `{"pc": "[3,0]", "status": "running"}` | VM | VM, pysdk |
| 指令坐标 | `[addr0, 0]` = opcode, `[addr0, -1]` = 读参数 | VM | VM |
| 命名槽位 | 局部变量的值（int/float/tensor 元信息） | VM, op-plat | VM, op-plat |
| 子栈 | CALL 产生的嵌套指令序列 | VM | VM |

```
/vthread/<vtid>                           → {pc, status}
/vthread/<vtid>/[0,0]                     → 操作码
/vthread/<vtid>/[0,-1]                    → 读参数 #1
/vthread/<vtid>/[0, 1]                    → 写参数 #1
/vthread/<vtid>/a                         → 命名槽位 (局部变量)
/vthread/<vtid>/mm                        → 命名槽位 (局部变量)
/vthread/<vtid>/[n,0]/[0,0]              → 子栈指令
```

**属性**: 机器执行层。指令展开为二维坐标 `[addr0, addr1]`，运行时零解析开销。

### 3.4 系统路径 `/sys/`

| 路径 | 内容 | 职责 |
|---|---|---|
| `/sys/vtid_counter` | int 计数器 | vthread ID 自增分配 |
| `/sys/config` | JSON | 全局配置 |
| `/sys/op-plat/<instance>` | JSON | op-plat 进程注册（存活性、负载） |
| `/sys/heap-plat/<instance>` | JSON | heap-plat 进程注册 |
| `/sys/vm/<id>` | JSON | VM 实例注册 |

### 3.5 命令队列 `/cmd/` `/notify/` `/done/`

使用 Redis List (FIFO)，是进程间异步通信的数据结构。

| 队列 | 生产者 | 消费者 | 用途 |
|---|---|---|---|
| `cmd:op-<backend>:<instance>` | VM | op-plat | 计算指令 |
| `cmd:heap-<backend>:<instance>` | VM | heap-plat | 生命周期指令 |
| `done:<vtid>` | op-plat, heap-plat | VM | 完成通知 |
| `notify:vm` | pysdk | VM | 新 vthread 唤醒 |

### 3.6 堆（隐式命名空间）

除保留路径外，KV 空间中的所有其他路径均为**堆变量**。按命名约定自行管理。

```
/models/bert/encoder_0/weights     → {"dtype":"f32", "shape":[768,3072], ...}
/data/cifar10/train                → {"dtype":"f32", "shape":[50000,3,32,32], ...}
/checkpoints/run_001/step_1000     → ...
```

---

## 4. Vthread — 执行流的数据结构

### 4.1 Vthread = 状态 + 栈 + 槽位

```
/vthread/<vtid>                 ← vthread 自身 (状态: pc + status)
/vthread/<vtid>/[0,0]           ← 指令坐标 (栈内指令序列)
/vthread/<vtid>/[0,-1]          ← 指令坐标 (读参数)
/vthread/<vtid>/[0, 1]          ← 指令坐标 (写参数)
/vthread/<vtid>/a               ← 命名槽位 (局部变量)
/vthread/<vtid>/mm              ← 命名槽位 (局部变量, ./mm 解析)
/vthread/<vtid>/[n,0]/[0,0]    ← 子栈 (CALL 产生)
```

**三个组成部分**：

| 组成部分 | 路径模式 | 存储内容 | 类比 |
|---|---|---|---|
| vthread 状态 | `/vthread/<vtid>` 本身的 value | `{"pc":..., "status":...}` | 线程控制块 (TCB) |
| 指令序列 | `/vthread/<vtid>/[addr0, addr1]` | 操作码 + 参数 | 栈中的指令 |
| 命名槽位 | `/vthread/<vtid>/<name>` | 变量值 (int/float/tensor元信息) | 栈中局部变量 |

### 4.2 二维寻址 `[addr0, addr1]`

指令坐标 `[addr0, addr1]` 是元程的核心寻址发明：

| addr1 的值 | 含义 | 示例 |
|---|---|---|
| `0` | 操作码 | `"matmul"`, `"call"`, `"add"` |
| `-1, -2, ...` | 读取参数（左值） | `"/models/A"`, `"./a"`, `"3.14"` |
| `1, 2, ...` | 写入参数（右值） | `"./Y"`, `"./out"` |

**设计意图**：
- addr0 是**序列维**：表示指令在栈帧中的顺序位置，PC 沿此维推进
- addr1 是**角色维**：将一条指令的操作码和参数展开为平级 key，VM 零解析读取

```
指令: matmul(A, B) -> ./Y
↓ 展开为执行层坐标:
/vthread/1/[0, 0] = "matmul"      ← 操作码
/vthread/1/[0,-1] = "/models/A"   ← 读参数 #1
/vthread/1/[0,-2] = "/models/B"   ← 读参数 #2
/vthread/1/[0, 1] = "./Y"         ← 写参数 #1
```

### 4.3 命名槽位与指令坐标的平级关系

命名槽位（`/vthread/<vtid>/<name>`）与指令坐标（`/vthread/<vtid>/[addr0, addr1]`）
是 `/vthread/<vtid>/` 下的**平级子 key**，互不嵌套：

```
/vthread/1/              ← vtid 自身（含 pc 和 status）
/vthread/1/[0,0]         ← 指令坐标
/vthread/1/[0,-1]        ← 指令坐标
/vthread/1/[0,1]         ← 指令坐标
/vthread/1/a             ← 命名槽位 (平级)
/vthread/1/mm            ← 命名槽位 (平级)
/vthread/1/[0,0]/[0,0]   ← 子栈 (嵌套，仅在 CALL 时存在)
```

**关键洞察**: 命名槽位不受 CALL/RETURN 嵌套影响。`./mm` 始终解析为 `/vthread/<vtid>/mm`，
无论当前处于第几层子栈。这使得 VM 在翻译编译层指令时不需要处理嵌套路径重写。

### 4.4 子栈 — CALL 的嵌套

```
根栈:           pc = "[0,0]"
根栈 CALL n 后: pc = "[n,0]/[0,0]"
子栈 CALL m 后: pc = "[n,0]/[m,0]/[0,0]"
```

子栈在路径上形成嵌套树，PC 字符串携带完整嵌套路径。
RETURN 时 DELETE 子栈子树 +PC 回到父栈。

### 4.5 Vthread 状态机

| 状态 | 含义 | PC 位置 |
|---|---|---|
| `init` | 已创建，待 VM 拾取 | 入口指令 |
| `running` | VM 正在调度执行 | 当前指令 |
| `wait` | 等待 op-plat / heap-plat 完成 | 已发射但未完成的指令 |
| `error` | 执行出错 | 出错指令 |
| `done` | 执行完毕 | 末尾 + 1 |

---

## 5. Func — 代码单元的数据结构

### 5.1 三层架构

函数有三种表示形式，分别服务于不同的角色：

| 层 | 位置 | 格式 | 角色 | 受众 |
|---|---|---|---|---|
| 源码层 | `/src/func/<name>` | dxlang 文本 | 人类可读、平台无关 | pysdk 写入、编译器读取 |
| 编译层 | `/op/<backend>/func/<name>` | dxlang 文本（融合/拆分后） | 后端优化后 | VM 读取执行 |
| 执行层 | `/vthread/<vtid>/` | `[addr0, addr1]` 二维坐标 | 机器直接执行 | VM |

```
数据流:
  pysdk → /src/func/             (源码)
            ↓ 编译器
  /op/op-cuda/func/              (CUDA 编译产物)
  /op/op-metal/func/             (Metal 编译产物)
            ↓ VM CALL 时 eager 翻译
  /vthread/<vtid>/[n,0]/...      (执行层)
```

### 5.2 指令格式

```
opcode(read_param_1, read_param_2, ...) -> write_param_1, write_param_2
```

`->` 左侧为输入（只读），右侧为输出（写入）。
参数可以是绝对路径 `"/models/A"`、相对路径 `"./a"`、或立即数 `"3.14"`。

### 5.3 控制流嵌套

```
/src/func/<name>/N               = "if" 或 "for"  ← 关键字
/src/func/<name>/N/cond          = 条件表达式
/src/func/<name>/N/true/0        = true 分支第 0 条
/src/func/<name>/N/false/0       = false 分支第 0 条
/src/func/<name>/N/body/0        = 循环体第 0 条
```

指令的嵌套结构与 Redis key 的嵌套结构**完全同构**。

---

## 6. Tensor — 数据的数据结构

### 6.1 Tensor 元信息

Tensor 的实际数据存于外部存储（POSIX shm、GPU 显存）。
KV 空间中仅存**元信息引用**：

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

| 字段 | 说明 |
|---|---|
| `dtype` | 数据类型：f32, f16, i32, ... |
| `shape` | 形状数组 |
| `byte_size` | 总字节数 = element_count × dtype_size |
| `device` | 所在设备：gpu0, gpu1, cpu |
| `address` | 物理地址信息 (shm 名称、机器标识) |
| `version` | 每次写入递增的版本号 |

### 6.2 Tensor 在系统中的流转

```
模型权重（堆）:
  /models/bert/weights    ← 长生命周期，全局共享

计算中间结果（栈）:
  /vthread/1/mm           ← 短生命周期，vthread 内可见

输入/输出（堆或栈）:
  取决于 pysdk 的用户设计
```

### 6.3 基础类型存储

对于非 Tensor 的简单值（int、float、bool），直接存为 value：

```
/vthread/1/a       = 1
/vthread/1/b       = 3.14
/vthread/1/flag    = true
```

---

## 7. 命令队列 — 通信数据结构

命令队列使用 FIFO List（如 Redis RPUSH/BLPOP），是进程间异步通信的核心机制：

```
生产者                  队列                       消费者
──────                  ────                       ──────
VM          → PUSH →  cmd:op-cuda:0    → RPOP →  op-cuda
VM          → PUSH →  cmd:heap-metal:0 → RPOP →  heap-metal
op-plat     → LPUSH → done:<vtid>      → BLPOP→  VM
pysdk       → LPUSH → notify:vm        → BLPOP→  VM
```

每条消息携带完整上下文（vtid, pc, 参数），消费者**无需回查**其他数据源即可执行。

### 7.1 op-plat 指令

```json
{
    "vtid": "1",
    "pc": "[3,0]",
    "opcode": "matmul",
    "inputs": [{"key": "...", "dtype": "f32", "address": {...}}],
    "outputs": [{"key": "...", "dtype": "f32", "address": {...}}],
    "params": {"transpose_a": false}
}
```

### 7.2 heap-plat 指令

```json
// newtensor
{"op": "newtensor", "key": "/models/weights", "dtype": "f32", "shape": [1024,512]}
// deltensor
{"op": "deltensor", "key": "/models/weights"}
// clonetensor
{"op": "clonetensor", "src": "/models/weights", "dst": "/models/weights_gpu1"}
```

### 7.3 完成通知

```json
{"pc": "[3,0]", "status": "ok", "outputs_updated": [{"key": "..."}]}
```

---

## 8. 锁 — 并发控制数据结构

```
/lock/<resource>   → holder 标识符, TTL 自动释放
```

使用场景：
- 保护堆 tensor 的并发修改: `/lock/tensor:/models/weights`
- 保护 vthread 拾取的原子性: VM 用 WATCH/MULTI/EXEC 事务

---

## 9. 数据结构关系总图

```
┌───────────────────────────────────────────────── KV 空间 ──────────────────────────────────────────────────┐
│                                                                                                             │
│  /src/func/                    /op/<backend>/func/              /vthread/<vtid>/                             │
│  ┌──────────────────┐         ┌───────────────────┐           ┌──────────────────────┐                      │
│  │ 函数签名 (text)   │         │ 编译后签名 (text)  │           │ {pc, status}          │                      │
│  │ func/gemm/0        │ 编译器  │ func/gemm/0        │ VM eager  │ [0,0] = "matmul"      │                      │
│  │ func/gemm/1        │ ──────→ │ ...                │ ────────→ │ [0,-1] = "/models/A"  │                      │
│  │ ...                │         │                    │  翻译     │ mm = tensor元信息      │                      │
│  └──────────────────┘         └───────────────────┘           │ /[n,0]/... (子栈)     │                      │
│                                                                 └──────────────────────┘                      │
│  /models/                      /cmd/ /notify/ /done/           /sys/                                         │
│  ┌──────────────────┐         ┌───────────────────┐           ┌──────────────────────┐                      │
│  │ 堆变量 (tensor)    │         │ 命令队列 (List)    │           │ 系统状态 (JSON)       │                      │
│  │ weights → {dtype,  │         │ cmd:op-cuda:0      │           │ vtid_counter = 42     │                      │
│  │  shape, shm}       │         │ done:<vtid>        │           │ op-plat/cuda:0 → {...}│                      │
│  │ data/cifar10 → ... │         │ notify:vm          │           │ heap-plat/metal:0→{..}│                      │
│  └──────────────────┘         └───────────────────┘           └──────────────────────┘                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 数据结构设计原则

| 原则 | 说明 | 体现 |
|---|---|---|
| **同构性** | 路径树的嵌套结构与语义层级一致 | 控制流嵌套 = key 嵌套 |
| **平级不嵌套** | 不同角色的数据不互相嵌套 | 命名槽位 ⊥ 指令坐标 |
| **自描述** | 每条数据携带完整上下文，消费者无需回查 | 命令队列消息包含 vtid+pc+完整参数 |
| **分离关注点** | 源码/编译/执行三层独立演进 | `/src/func/` ⊥ `/op/.../func/` ⊥ `/vthread/` |
| **值大小约束** | 大对象不存 KV 空间，仅存引用 | Tensor 数据存 shm，KV 存 meta |
| **异步优先** | 所有跨进程通信通过队列，天然异步 | 命令队列 + 完成通知 |

---

> **关联文档**:
> - [spec-v1.md](spec-v1.md) — 元程规范 v1（抽象模型）
> - [deepx-design.md](deepx-design.md) — DeepX 实现设计（五个核心）
> - [redis-keys.md](redis-keys.md) — Redis Key 设计速查表
