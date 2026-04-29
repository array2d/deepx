# 元程 — 数据结构

> 定义元程的基础数据类型、扩展数据类型，以及在 KV 空间中的布局。

---

## 1. 为什么单列一篇数据结构

C 语言的数据结构定义在 `<stdint.h>` 和 `struct` 中，编译器在编译期确定每个变量的
字节偏移。CUDA 继承了 C 的类型体系，加上 `__shared__` / `__constant__` 内存空间标注。
这些类型定义是语言的基石——写了 `int x`，编译器就知道分配 4 字节。

元程没有编译器。数据类型不是编译器规则，而是**所有进程之间共享的约定**。
基础类型（int / float / bool / string）直接存为 KV 空间的 value。
扩展类型（tensor）的 value 是元信息 JSON，实际数据通过 shm 指针引用。
KV 空间的路径布局就是元程的"内存映射"。

---

## 2. 基础数据类型

元程的基础类型直接以字面量或短字符串形式存储在 KV value 中，无需额外编码：

| 类型 | 示例 value | value 大小 | 对应 C 类型 | 对应 CUDA |
|------|-----------|-----------|------------|----------|
| `int` | `1`, `-42`, `0` | ~1-11 字节（字符串） | `int` (4B) | `int` |
| `float` | `3.14`, `-0.5`, `1e-5` | ~4-15 字节 | `float` (4B) / `double` (8B) | `float` / `double` |
| `bool` | `true` / `false` | 4-5 字节 | `bool` (1B，通常) | `bool` |
| `string` | `"hello"`, `"f32"` | 取决于内容，建议 < 1KB | `char[]` / `char*` | `char*` |

**基础类型可以直接作为 KV 的 value**：

```
/vthread/1/flag    = true        ← bool
/vthread/1/count   = 42          ← int
/vthread/1/rate    = 0.001       ← float
/vthread/1/name    = "f32"       ← string
```

与 C 的关键区别：

| | C / CUDA | 元程 KV |
|---|---|---|
| 存储位置 | 栈 / 寄存器 / 显存 | Redis key 的 value |
| 寻址方式 | 指针 / 偏移量 | 字符串路径 |
| 类型检查 | 编译期 | 运行时（VM 解析 value） |
| 生命周期 | 作用域退出即释放 | 显式 DELETE |
| 精度控制 | `int` vs `int32_t` 编译器决定 | 约定决定（建议标注 dtype string） |

**注意**：基础类型的 value 在 Redis 中存为字符串。Redis 的 int 操作（INCR）可用，
但建议统一走 GET→解析→运算→SET 路径以保证类型安全。

---

## 3. 扩展数据类型：Tensor

### 3.1 定义

Tensor 是元程唯一的一级扩展数据类型。它是一个多维数组，其实际数据存储于外部内存
（POSIX shm / GPU 显存），KV 空间中仅存储**元信息引用**：

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
| `dtype` | string | 是 | `f16`, `f32`, `f64`, `bf16`, `bf8`, `i8`, `i16`, `i32`, `i64`, `u8`, `bool` |
| `shape` | array[int] | 是 | 如 `[1024, 512]`, `[2, 3, 224, 224]` |
| `byte_size` | int | 是 | element_count × dtype_size。对于 bool，1 bit → 向上取整到字节 |
| `device` | string | 是 | `gpu0`, `gpu1`, `cpu` |
| `address` | object | 是 | 物理地址信息 |
| `address.node` | string | 是 | 机器标识，如 `"n1"` |
| `address.type` | string | 是 | `"shm"` (POSIX共享内存) 或 `"gpu"` (GPU显存直接引用) |
| `address.shm_name` | string | 是 | POSIX shm 名称，如 `"/deepx_t_abc123"` |
| `ctime` | int | 否 | 创建时间戳 |
| `version` | int | 否 | 每次写入后递增 |

### 3.2 dtype 字节宽度

| dtype | 字节数 | C/CUDA 对应 | 说明 |
|-------|--------|-----------|------|
| `f16` | 2 | `half` / `__half` | IEEE 754 half |
| `f32` | 4 | `float` | IEEE 754 single |
| `f64` | 8 | `double` | IEEE 754 double |
| `bf16` | 2 | `__nv_bfloat16` | Brain floating point |
| `bf8` | 1 | — | 8-bit brain float (E4M3 / E5M2) |
| `i8` | 1 | `int8_t` / `char` | |
| `i16` | 2 | `int16_t` / `short` | |
| `i32` | 4 | `int` / `int32_t` | |
| `i64` | 8 | `long long` / `int64_t` | |
| `u8` | 1 | `uint8_t` | |
| `bool` | 1 | `bool` | 每元素 1 bit，byte_size = ceil(n/8) |

```
byte_size 计算:
  element_count = prod(shape)                                    // shape 各维乘积
  byte_size     = element_count * dtype_size                     // 普通类型
  byte_size     = ceil(element_count / 8)                        // bool 类型
```

### 3.3 与 C / CUDA 的数据结构对比

#### 3.3.1 数组声明对比

```
C:
  float A[1024][512];                      ← 编译期确定大小，栈或数据段
  float* A = malloc(1024 * 512 * 4);       ← 运行时分配，堆

CUDA:
  float* d_A;
  cudaMalloc(&d_A, 1024 * 512 * 4);        ← 显存分配
  __shared__ float s_A[256];               ← 共享内存

元程:
  GET /models/A → {dtype:"f32", shape:[1024,512], address:{shm_name:"/deepx_t_xxx"}}
                                              ← KV 中存元信息
  shm_open("/deepx_t_xxx") → ptr              ← 实际数据在 shm 中
```

#### 3.3.2 核心差异

| 维度 | C | CUDA | 元程 |
|------|---|------|------|
| **数据位置** | 栈/堆(CPU 内存) | global/shared/constant 显存 | POSIX shm / GPU 显存（由 device 字段决定） |
| **寻址方式** | 指针 `float*` + 偏移 | 指针 `float*` + 偏移 | KV 路径 `GET /models/A` → 元信息 → shm ptr |
| **形状信息** | 丢失（退化为指针） | 丢失（退化为指针） | **保留**（shape 字段始终可查） |
| **类型信息** | 编译期已知，运行期丢失 | 同 C | **保留**（dtype 字段始终可查） |
| **跨进程传递** | 需序列化 + IPC | 需 cudaIpcOpenMemHandle | 路径字符串 + shm name（元程原生） |
| **生命周期** | 作用域/手动 free | 手动 cudaFree | heap-plat 管理 newtensor/deltensor |
| **多设备** | 不涉及 | 单 GPU，显式 cudaMemcpy | 元信息标注 device，heap-plat 负责跨设备克隆 |
| **并发安全** | 锁 | 原子操作 + 同步 | Redis LOCK / WATCH + 原子操作 |

#### 3.3.3 形状/类型不丢失的设计收益

C 和 CUDA 中，数组一旦传给函数就退化为指针，丢失 shape 和类型信息：

```c
// C: 函数签名看不到 shape
void matmul(float* A, float* B, float* C, int M, int N, int K);
//          ↑ 仅指针            ↑ shape 靠额外参数传递

// CUDA: 同样问题，加上显存地址语义
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
```

元程中，shape 和 dtype 是 tensor 元信息的**一等字段**，不会丢失：

```
op-plat 收到的指令:
{
  "opcode": "matmul",
  "inputs": [
    {"key": "/models/A", "dtype": "f32", "shape": [1024, 512], "address": {...}},
    {"key": "/models/B", "dtype": "f32", "shape": [512, 256],  "address": {...}}
  ],
  "outputs": [
    {"key": "/vthread/1/c", "dtype": "f32", "shape": [1024, 256], "address": {...}}
  ]
}
```

op-plat 从指令中直接获取 shape 和 dtype，无需额外查询。这消除了 C/CUDA 中
"传指针的同时还要传 shape 参数"的惯例。

---

## 4. KV 空间中的类型布局

### 4.1 概念上的"内存映射"

KV 空间的路径不是线性的字节地址，而是一棵键值树。不同类型的值分布在不同的子树中：

```
/ (根)
│
├── 基础类型存放在 vthread 栈或系统路径中，value 是字面量
├── Tensor 元信息分布在堆路径和 vthread 栈中，value 是 JSON
└── Tensor 实际数据不存 KV 空间，通过 shm_name 引用
```

### 4.2 各路径区间的类型分布

| 路径区间 | 存放类型 | value 形式 | 读写者 |
|----------|---------|-----------|--------|
| `/vthread/<vtid>/<name>` | **基础类型** 或 **tensor 元信息** | 字面量 或 JSON | VM, op-plat |
| `/vthread/<vtid>` | **JSON**（状态对象） | `{"pc":"...","status":"..."}` | VM（写），pysdk（读） |
| `/vthread/<vtid>/[addr0, addr1]` | **string**（指令片段） | `"matmul"`, `"./a"`, `"3.14"` | VM（读写） |
| `/models/*`, `/data/*` | **tensor 元信息** | JSON | heap-plat（写），VM/op-plat（读） |
| `/src/func/<name>/N` | **string**（指令文本） | `"add(A, B) -> ./C"` | pysdk（写），VM（读） |
| `/op/<backend>/func/<name>/N` | **string**（编译后指令） | 同上 | 编译器（写），VM（读） |
| `/sys/*` | **基础类型** 或 **JSON** | int 或 JSON | 各进程（注册/读取） |
| `/cmd/*`, `/done/*`, `/notify/*` | **JSON**（消息） | 见下方 | 各进程 |

### 4.3 具体布局示例

```
# === 基础类型 ===
/vthread/1/flag        = true                      ← bool, 直接存
/vthread/1/iter        = 0                         ← int, 直接存
/vthread/1/lr          = 0.001                     ← float, 直接存
/sys/vtid_counter      = 42                        ← int, 直接存 (Redis INCR 可用)

# === Tensor 元信息 (JSON) ===
/models/bert/W         = {"dtype":"f32","shape":[768,3072],"device":"gpu0","address":{...},"byte_size":...}
/models/bert/b         = {"dtype":"f32","shape":[3072],"device":"gpu0","address":{...},"byte_size":...}
/vthread/1/mm          = {"dtype":"f32","shape":[1024,256],"device":"gpu0","address":{...},"byte_size":...}
/vthread/1/out         = {"dtype":"f32","shape":[1024,10],"device":"cpu","address":{...},"byte_size":...}

# === 指令 (string) ===
/src/func/gemm/0       = "matmul(A, B) -> ./Y"
/vthread/1/[0,0]       = "matmul"
/vthread/1/[0,-1]      = "/models/A"
/vthread/1/[0,-2]      = "/models/B"
/vthread/1/[0, 1]      = "./Y"

# === vthread 状态 (JSON) ===
/vthread/1             = {"pc":"[0,0]","status":"running"}

# === 系统注册 (JSON) ===
/sys/op-plat/cuda:0    = {"program":"op-cuda","device":"gpu0","status":"running","load":0.3}
/sys/config            = {"max_vthreads":100,"timeout_ms":30000}

# === 命令消息 (JSON, 存储在 List 中) ===
cmd:op-cuda:0          ← RPUSH {"vtid":"1","pc":"[0,0]","opcode":"matmul","inputs":[...],"outputs":[...]}
done:1                 ← LPUSH {"pc":"[0,0]","status":"ok"}
```

### 4.4 值大小约束

| 类型 | 最大 value 大小 | 原因 |
|------|----------------|------|
| 基础类型 (int/float/bool) | ~15 字节 | 字面量字符串 |
| string | 1 KB | 避免 Redis 大 key 性能问题 |
| Tensor 元信息 (JSON) | ~10 KB | 主要是 address 字段，shm_name 通常 < 64B |
| Tensor 实际数据 | **不存 KV** | 存外部 shm/显存，KV 仅存引用 |
| 命令消息 (JSON) | ~10 KB | 包含 tensor 元信息的完整拷贝 |

**设计原则**：KV 空间存**描述**，外部存储存**数据**。这是元程与 C/CUDA 的又一
根本差异——C 的 struct 是数据本身，元程的 value 是数据的**引用和描述**。

---

## 5. Tensor 元信息 vs C struct vs CUDA 内存描述符

### 5.1 三方对照

```
C struct (在 CPU 内存中表示一个数组):
  struct {
      float* data;          ← 指向实际数据
      int dims[4];          ← 形状（可选，取决于库）
      int ndim;             ← 维度数（可选）
      int dtype;            ← 类型枚举（可选，通常编译期已知）
  }
  问题: 每个库自己定义 (PyTorch ATen, NumPy, Eigen 各不同)

CUDA 内存描述符:
  struct cudaPointerAttributes {
      enum cudaMemoryType memoryType;    ← host / device / managed
      int device;                        ← GPU 编号
      void* devicePointer;               ← 设备指针
      void* hostPointer;                 ← 主机指针（统一内存时）
  }
  问题: 没有 shape / dtype，仅描述"指针属性"，不是"数组属性"

元程 Tensor 元信息 (JSON, 存于 KV):
  {
      "dtype": "f32",                    ← 类型始终可查
      "shape": [1024, 512],              ← 形状始终可查
      "byte_size": 2097152,              ← 预计算，O(1) 获取
      "device": "gpu0",                  ← 设备标注
      "address": {
          "node": "n1",                  ← 机器
          "type": "shm",                 ← 内存类型
          "shm_name": "/deepx_t_xxx"     ← 全局唯一标识
      },
      "version": 5                       ← 变更追踪
  }
  优势: 自包含、跨语言、跨进程、携带完整描述
```

### 5.2 跨设备传递对比

```
场景: GPU0 上的 tensor 需要传给 GPU1

CUDA:
  cudaMemcpyPeer(dst, 1, src, 0, size);    ← 显式 P2P 拷贝
  // 需要知道: 源指针, 目标指针, 大小, 两个 GPU 编号
  // 调用者自己管理这些信息

元程:
  1. VM 发送 clonetensor 到 heap-plat:
     {"op":"clonetensor", "src":"/models/W", "dst":"/models/W_gpu1", "device":"gpu1"}
  2. heap-plat GET /models/W → 获取完整元信息 (含当前 device)
  3. heap-plat 在 gpu1 上分配 shm → memcpy → 写入新元信息
  4. 完成

  // VM 只需要知道路径和目标 device，heap-plat 处理其余全部
```

---

## 6. 类型安全与运行时检查

C/CUDA 的类型检查在编译期，运行时只有原始字节。元程的类型检查在运行时：

| | C / CUDA | 元程 |
|---|---|---|
| 检查时机 | 编译期 `float* p = (float*)malloc(...)` | 运行时 op-plat 解析 dtype 字段 |
| 类型错误后果 | 编译失败（安全）或未定义行为（cast） | op-plat 返回 error: `"dtype mismatch: f32 vs f16"` |
| shape 不匹配 | 段错误 / 静默错误 | op-plat 返回 error: `"shape mismatch: [10] vs [20]"` |
| 内存越界 | 段错误 / 安全漏洞 | shm 大小由 byte_size 约束，op-plat 可做边界检查 |

---

## 7. 汇总：元程数据类型体系

```
元程数据类型
├── 基础类型 (value = 字面量)
│   ├── int       → "42"
│   ├── float     → "3.14"
│   ├── bool      → "true"
│   └── string    → "hello"
│
├── 扩展类型 (value = JSON 元信息，数据在外部)
│   └── tensor
│       ├── dtype    : f16 | f32 | f64 | bf16 | bf8 | i8 | i16 | i32 | i64 | u8 | bool
│       ├── shape    : [d1, d2, ...]
│       ├── byte_size: element_count × dtype_size
│       ├── device   : gpu0 | gpu1 | cpu
│       └── address  : {node, type, shm_name}
│
└── 结构化类型 (value = JSON 对象，数据在 KV 中)
    ├── vthread 状态  : {pc, status, error?}
    ├── 进程注册      : {program, device, status, load}
    ├── 算子元数据    : {category, dtype, max_shape, replaces?}
    └── 命令消息      : {vtid, pc, opcode, inputs, outputs}
```

---

> **关联文档**:
> - [README.md](README.md) — 元程总篇（核心思想：程序=数据结构+函数+数据）
> - [spec-v1.md](spec-v1.md) — 元程规范 v1（抽象模型）
> - [redis-keys.md](redis-keys.md) — Redis Key 布局速查表
