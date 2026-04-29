# 元程 (Metaproc)

> **程序 = 数据结构 + 函数 + 数据**
>
> 这是元程的核心思想。在分布式系统中，三者被**显式地、可见地、共享地**定义在
> 一个全局 KV 空间中。数据结构是地址空间的划分规则，函数是可复用的代码单元，
> 数据是计算的实际内容。三者分离，各司其职，通过路径空间统一编排。

---

## 1. 为什么是"元程"

### 1.1 问题的根源

在单机上写程序，OS 和编译器已经替我们管理好了一切：

```
程序员的视角 (单机):
  x = a + b          ← 看起来只有"数据"和"运算"
  y = f(x)
  return y

OS/编译器实际做的 (对程序员透明):
  - 在栈上为 x 分配 4 字节
  - 为 a 找到寄存器 r1
  - call f → 分配新栈帧 → 保存 rbp → 传递参数 → ret → 恢复 rbp
  - 所有这些（数据结构）都是隐式的
```

但在分布式系统中，**隐式不再可行**。因为：
- 计算分布在不同的 GPU/CPU 上
- 数据存储在不同的 shm/显存/磁盘上
- 进程之间只能通过约定的数据格式通信
- 没有"OS"替你管理全局状态

**你需要一个显式的、全局可见的、所有进程都能读懂的数据结构约定。**
这就是元程的 KV 空间。

### 1.2 "元程"的含义

"元程"（Metaproc）是"元进程"的缩写：

- **元 (Meta)**：它定义了一个"进程"的抽象骨架——不是运行某个具体程序的进程，
  而是可以运行**任意程序**的进程框架。类比：OS 内核定义了进程的抽象
  （地址空间、线程、栈、堆），但内核本身不规定进程里跑什么程序。
- **程 (Proc)**：它本身是一个"进程"——具有地址空间（KV 空间）、
  执行流（vthread）、代码段（func）、数据段（heap），遵循与 OS 进程相同的
  抽象模式。

```
OS 进程 = 虚拟地址空间 + 线程 + 代码段 + 数据段 + 栈
元程    = KV 空间        + vthread + /src/func/ + 堆    + /vthread/
```

**关键区别**：OS 进程的地址空间是私有的、字节寻址的、单机的。
元程的 KV 空间是**共享的、路径寻址的、分布式的**。

---

## 2. 核心公式：程序 = 数据结构 + 函数 + 数据

### 2.1 三元分解

```
程序 = 数据结构 + 函数 + 数据

数据结构：变量的容器、寻址方式、组织规则
          → KV 路径空间 + 保留路径约定 + 二维寻址

函数：    可复用的执行逻辑单元
          → /src/func/ + /op/<backend>/func/（源码层 + 编译层）

数据：    实际被处理和传输的内容
          → 堆 tensor + 栈局部变量 + 立即数
```

### 2.2 三元在元程中的映射

| | 数据结构 | 函数 | 数据 |
|---|---|---|---|
| **是什么** | KV 空间 + 路径约定 | func 定义 | tensor + 基础类型 |
| **存储位置** | 路径本身的组织 | `/src/func/`, `/op/.../func/` | `/models/`, `/vthread/<vtid>/<name>` |
| **生命周期** | 系统级别的常量 | 注册后长期有效 | 取决于类型（堆=长期，栈=短期） |
| **谁定义** | 元程规范 | pysdk（用户代码） | pysdk（用户数据） |
| **谁使用** | 所有进程 | VM + 编译器 | op-plat + heap-plat |
| **类比 OS** | 虚拟地址空间布局 | .text 段 + 共享库 | .data/.bss + 栈 |

### 2.3 为什么三元必须分离

```
反例：如果把数据和函数混在一起
  /vthread/1/data/weights = ...
  /vthread/1/func/forward = ...
  → 每个 vthread 都要复制一份代码 → 浪费
  → 函数和数据的生命周期不同 → 管理混乱
  → VM 无法跨 vthread 复用代码 → 不能 CALL

正解：三元分离
  函数: /src/func/forward        ← 全局一份，所有 vthread CALL
  数据: /models/weights           ← 全局共享，vthread 通过路径引用
  结构: /vthread/1/               ← 每个 vthread 独立，栈 + 局部变量
```

---

## 3. 元程的五个核心

元程系统由五个核心组件构成：

| 核心 | 角色 | 是什么 | 做什么 |
|---|---|---|---|
| **KV 空间** | 全局状态存储 | 数据结构 | 路径空间、命令队列、锁 |
| **pysdk** | 算法前端 | 函数 + 数据的写入者 | 注册 func 源码、创建 vthread |
| **op-plat** | 计算后端 | 函数的执行者 | 被动消费指令、执行 GPU/CPU 张量运算 |
| **heap-plat** | 堆管理 | 数据的生命周期管理 | tensor 创建/删除/克隆 |
| **VM** | 解释执行 | 数据结构的调度者 | func 翻译、指令路由、vthread 状态推进 |

```
┌─ pysdk ────────────────────────────────────────────────┐
│  函数: 注册源代码到 /src/func/                           │
│  数据: 通过 heap-plat 创建 tensor                       │
│  结构: 创建 vthread 到 /vthread/                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─ KV 空间 ───────────────────────────────────────────────┐
│  /src/func/  源码  │  /op/.../func/  编译  │  /vthread/  执行  │
│  堆变量           │  命令队列             │  锁                  │
└──┬────────────┬──────────────┬────────────────────────────┘
   │            │              │
   ▼            ▼              ▼
┌──────┐  ┌──────────┐  ┌───────────┐
│  VM  │  │ op-plat  │  │ heap-plat │
│      │  │          │  │           │
│ 解释 │  │ 执行张量  │  │ 管理tensor│
│ 调度 │  │ 计算      │  │ 生命周期  │
└──────┘  └──────────┘  └───────────┘
```

### 3.1 进程的被动性

所有核心进程（VM、op-plat、heap-plat）都**被动**执行——它们不主动发起操作，
而是消费命令队列。这类似于 OS 中的中断驱动模型：

| 进程 | 等待什么 | 谁唤醒它 |
|---|---|---|
| VM | `BLPOP notify:vm` + `BLPOP done:<vtid>` | pysdk + op-plat/heap-plat |
| op-plat | `RPOP cmd:op-<backend>:<instance>` | VM |
| heap-plat | `RPOP cmd:heap-<backend>:<instance>` | VM |

**为什么被动？**
- 被动 = 解耦：生产者不需要知道消费者的具体实例
- 被动 = 可伸缩：同一个命令队列可以有多个消费者竞争
- 被动 = 容错：消费者崩溃，消息仍在队列中，重启后继续消费

---

## 4. 执行全景

### 4.1 从用户代码到 GPU 执行

```
用户 Python 代码:
  y = x @ w + b
  y = y.relu()

        ↓ pysdk 翻译

/src/func/forward:
  /src/func/forward/0 = matmul(./x, /models/W) -> ./mm
  /src/func/forward/1 = add(./mm, /models/b) -> ./mm
  /src/func/forward/2 = relu(./mm) -> ./y

        ↓ 编译器优化（可选）

/op/op-cuda/func/forward:
  /op/op-cuda/func/forward/0 = fused_matmul_add_relu(./x, /models/W, /models/b) -> ./y

        ↓ VM CALL → eager translate

/vthread/1/[0,0]:
  /vthread/1/[0,0]/[0,0] = "fused_matmul_add_relu"    ← 操作码
  /vthread/1/[0,0]/[0,-1] = "./x"                     ← 读参数
  /vthread/1/[0,0]/[0,-2] = "/models/W"
  /vthread/1/[0,0]/[0,-3] = "/models/b"
  /vthread/1/[0,0]/[0, 1] = "./y"                     ← 写参数

        ↓ VM dispatch → PUSH 到 op-plat

cmd:op-cuda:0 ← {"vtid":"1", "opcode":"fused_matmul_add_relu", ...}

        ↓ op-plat 消费

op-cuda: RPOP → 解析参数 → GET tensor 元信息 → shm_open → GPU kernel → 完成

        ↓ 完成通知

done:1 ← {"pc":"[0,0]", "status":"ok"}

        ↓ VM 醒来

/vthread/1 = {"pc": "[1,0]", "status": "done"}  ← 继续下一条
```

### 4.2 控制流示例

```
func dynamic_forward(x, threshold) -> (y) {
    sum(./x) -> ./s
    greater(./s, threshold) -> ./cond

    if (./cond) {
        add(./x, 1.0) -> ./y
    } else {
        mul(./x, -1.0) -> ./y
    }                               ← VM 直接解释 if，不经过 op-plat
}
```

VM 在处理控制流指令时**不需要 PUSH 到 op-plat**——它自己就是控制流的解释器。

---

## 5. 与 OS 进程模型的对照

元程的设计有意模仿了 OS 进程的抽象，但在分布式维度上重新定义了每个概念：

| OS 进程概念 | 元程对应 | 关键区别 |
|---|---|---|
| 虚拟地址空间 | KV 空间 | 字符串路径寻址、全局共享、分布式 |
| 线程 | Vthread | 状态存储于 KV 空间，多 VM 可并行拾取 |
| 代码段 (.text) | `/src/func/` + `/op/.../func/` | 人类可读 dxlang、后端可优化 |
| 堆段 (.data/.bss) | 非保留路径（堆变量） | tensor 元信息存 KV，实际数据存 shm |
| 栈段 | `/vthread/<vtid>/` 子树 | CALL 产生子栈嵌套，RETURN 删除 |
| 程序计数器 (PC) | `pc` 字段：`"[3,0]"` 或 `"[3,0]/[0,0]"` | 字符串路径，天然嵌套 |
| 栈帧 | CALL 产生的子维度 | 二维坐标 `[addr0, addr1]` |
| 系统调用 | PUSH 到 op-plat / heap-plat 队列 | 异步、可并行、可批量 |
| 互斥锁 | `LOCK/UNLOCK` | TTL 自动释放 |

### 5.1 与 x86 指令的对照

| x86 指令 | 元程等价 | 执行者 |
|---|---|---|
| `mov [addr], value` | `SET /vthread/1/x = value` | VM |
| `add eax, ebx` | `PUSH add(./a, ./b) -> ./c` 到 op-plat | op-plat |
| `call func` | VM 翻译 `/op/.../func/` → `eager inline` 到子栈 | VM |
| `ret` | DELETE 子栈 + PC 回父栈 | VM |
| `cmp + jcc` | `if (./cond)` → VM 直接分支 | VM |
| `int 0x80` (syscall) | PUSH 到 cmd 队列 | VM→op-plat/heap-plat |

---

## 6. 元程的"元"

### 6.1 元程不是一种语言

元程是一个**分布式计算模型**。它不规定：
- 前端语言（Python、Go、C++ 都可以）
- 后端硬件（CUDA、Metal、CPU 都可以）
- 存储实现（Redis、etcd、自研 KV 都可以）

它只规定：
- 数据结构怎么组织（KV 路径空间 + 保留路径约定）
- 函数怎么定义（`->`/`<-` 读写分离 + dxlang 指令格式）
- 数据怎么流转（命令队列 + 完成通知）
- 执行流怎么管理（vthread 状态机 + CALL/RETURN）

### 6.2 元程满足的条件

只要一个 KV 存储提供以下能力，就能运行元程程序：

| 能力 | 说明 |
|---|---|
| `GET/SET/DELETE` | 基本 KV 操作 |
| List (FIFO 队列) | `PUSH/POP/BLPOP` |
| 锁 | `LOCK/UNLOCK` + TTL |
| 事务 | `WATCH/MULTI/EXEC` 原子操作 |
| 基础类型 | int, float, bool, string, JSON |

当前参考实现：DeepX，使用 Redis 作为 KV 空间。

---

## 7. 文档索引

| 文档 | 说明 |
|---|---|
| **[spec-v1.md](spec-v1.md)** | 元程规范 v1 — 抽象模型（KV 空间要求、vthread 模型、CALL/RETURN、异步执行） |
| **[metaproc-datastruct.md](metaproc-datastruct.md)** | 数据结构篇 — KV 路径空间、保留路径、vthread、func、tensor、命令队列 |
| **[deepx-design.md](deepx-design.md)** | DeepX 实现设计 — 五个核心组件的具体实现（Redis key 约定、VM 执行循环、op-plat/heap-plat 协议） |
| **[redis-keys.md](redis-keys.md)** | Redis Key 设计速查表 — 所有路径的 value 类型与示例 |
| **[dev-heap-plat.md](dev-heap-plat.md)** | heap-plat 开发指南 |
| **[dev-op-plat.md](dev-op-plat.md)** | op-plat 开发指南 |
| **[dev-pysdk.md](dev-pysdk.md)** | pysdk 开发指南 |

### 7.1 阅读顺序建议

```
第一遍（理解元程是什么）:
  1. 本文 (README.md)                    ← 核心思想
  2. metaproc-datastruct.md              ← 数据结构全貌
  3. spec-v1.md                          ← 抽象规范

第二遍（理解 DeepX 怎么实现）:
  4. deepx-design.md                     ← 五个核心实现
  5. redis-keys.md                       ← 路径速查

第三遍（上手开发）:
  6. dev-heap-plat.md / dev-op-plat.md / dev-pysdk.md
```

---

## 8. 名称的由来

**Metaproc** = Meta + Process

- **Meta**（元）：它是"进程的进程"——定义了进程的抽象框架，而不绑定具体程序。
  就像 OS 定义了进程的抽象但不规定进程里跑什么，元程定义了分布式进程的抽象
  但不规定具体算法。
- **Proc**（程）：它本身具有进程的全部特征——地址空间、执行流、代码段、数据段、
  栈。只是这些概念被重新定义为分布式版本。

与"元编程"（Metaprogramming）的区别：
- 元编程：程序操作程序（生成代码、反射、宏）
- 元程：定义程序的程序（定义分布式进程的抽象框架）

---

> **核心命题**：在一个多机、多 GPU 的分布式系统中，你不能依赖 OS 替你管理状态。
> 你必须显式定义：数据结构是什么（KV 路径空间）、函数放在哪（`/src/func/`）、
> 数据怎么传（命令队列 + 完成通知）、执行流怎么管理（vthread + PC）。
> 元程就是这三元的统一框架。
