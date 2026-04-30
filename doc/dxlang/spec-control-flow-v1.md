# DeepX 控制流与前端代码生成 — 架构分析 v1

> 方案 C 确定为最优架构方向。本文展开 5 种子方案 + deepxir 语法完整设计。

---

## 1. 评判准则

| 准则 | 含义 |
|------|------|
| **关注点分离** | 每层只做一件事，层间接口明确 |
| **单一抽象级别** | 同一层内的概念在同一抽象高度 |
| **可验证性** | 层间转换有形式化保证 |
| **可扩展性** | 新能力不触动核心 |
| **概念完整性** | 一个核心思想贯穿始终 |

---

## 2. 方案 C 的本质

> **控制流从"用户意图"到"机器执行"是一个语义等价的格式变换过程。
> 这个变换应该发生在明确的边界上，每步变换可独立验证。**

```
                    IR 层数          Lowering 位置
C1 单层关键字        1 层            无（VM 直接解释）
C2 二层分离          2 层            Scheduler 服务
C3 单层基本块        1 层            前端负责
C4 单层 Region       1 层            VM 原生执行 region
C5 关键字+VM内lower  1 层（外部）    VM 加载时内部 lowering
```

---

## 3. deepxir 语法设计

> 语法需同时满足三者：**人**（可扫读）、**VM**（确定可解析）、**Agent**（模式固定可生成）。

### 3.1 核心约束

| 约束 | 原因 |
|------|------|
| **保留 `->` 和 `<-`** | 显式区分只读参数和写入参数方向，是 deepxir 的协议级语义 |
| **算子统一命名** | `tensor.new` / `tensor.del` 形成命名空间；栈变量 VM 自动推导 |
| **中缀运算符需 op-plat 注册** | 非标量类型的 `+`/`-`/`*`/`>` 等符号运算，后端必须声明支持 |

### 3.2 语法规则

```
program     = func*
func        = "fn" name "(" params ")" ("->" returns)? body
params      = param ("," param)*
param       = name (":" type)?
returns     = name (":" type)?
            | "(" name (":" type)? ("," name (":" type)?)* ")"
type        = "f16" | "f32" | "f64" | "bf16" | "i8" | "i16" | "i32" | "i64" | "bool" | "string"
            | type "[" shape "]"           # f32[2,4] 或 f32[?,?]
shape       = dim ("," dim)*
dim         = int | "?"

body        = "{" stmt* "}"
stmt        = assign  |  ctrl_if  |  ctrl_for  |  ctrl_while
            | ctrl_break  |  ctrl_continue  |  ctrl_switch  |  ctrl_return
            | lifecycle  |  bare_expr

# === 赋值：两种箭头，语义等价 ===
assign      = prefix_op "->" name          # 传统: 表达式在左, 结果在右
            | name "<-" prefix_op          # C风格: 结果在左, 表达式在右
            | infix_expr "->" name         # 中缀: x + y -> z
            | name "<-" infix_expr         # C中缀: z <- x + y

# === 前缀调用（总是合法）===
prefix_op   = name "(" args ")"            # add(x, y), matmul(A, B)
            | unop operand                 # !flag, -x

# === 中缀表达式（仅当 op-plat 注册了该符号）===
infix_expr  = operand binop operand        # x + y, s > 0
binop       = "+" | "-" | "*" | "/" | "%"
            | "==" | "!=" | "<" | ">" | "<=" | ">="
            | "&&" | "||"
unop        = "-" | "!"

# === 操作数 ===
operand     = name                          # 局部变量: x, tmp
            | "/" name ("/" name)*         # 堆路径: /models/W
            | literal                       # 1.0, true, "f32"
args        = operand ("," operand)*
literal     = int | float | "true" | "false" | string

# === 生命周期 ===
lifecycle   = "tensor.new" "(" shape "," type ")" "->" name    # 堆分配
            | "tensor.del" "(" name ")"                        # 堆释放
            | "tensor.clone" "(" name ")" "->" name            # 堆克隆

# 栈变量: VM 遇到新 name 在写位置首次出现时自动创建, 无需显式 tensor.new

# === 控制流 ===
ctrl_if     = "if" operand body ("else" (ctrl_if | body))?
ctrl_for    = "for" name "in" range body
range       = operand ".." operand
ctrl_while  = "while" operand body
ctrl_loop   = "loop" body
ctrl_break  = "break"
ctrl_continue = "continue"
ctrl_switch = "switch" operand "{" case* default? "}"
case        = "case" literal ":" body
default     = "default" ":" body
ctrl_return = "ret" operand?
```

### 3.3 语法决策说明

#### 为什么必须保留 `->` 和 `<-`

deepxir 不是通用编程语言。它的指令格式（`reads` 数组 + `writes` 数组）天然有方向性。
箭头直接表达这个方向：

```
add(x, y) -> z        reads=[x, y]  writes=[z]    一目了然
z <- add(x, y)        reads=[x, y]  writes=[z]    C 风格等价写法

# 如果用 =，方向信息丢失:
z = add(x, y)         ← 失去了 reads/writes 的结构区分
```

`->` 和 `<-` 是**语法级的多写入支持**：

```
split(x) -> (a, b)       # 一目了然：x 读, a 和 b 写
(a, b) <- split(x)       # 等价 C 风格

# 如果用 = 则需要特殊语法:
(a, b) = split(x)        # 混淆了 assignment 和 destructure
```

#### 为什么中缀需要 op-plat 注册

```deepxir
# 场景 1: VM 原生求值（标量）
1 + 2 -> x            # VM 直接算, 不需要 op-plat

# 场景 2: op-plat 求值（张量）
x + y -> z            # 仅当 op-plat 注册了 add 算子支持 "+" 符号时合法
                      # 否则编译/解析时报错, 强制使用显式前缀:
add(x, y) -> z

# 场景 3: 混合
x + 1.0 -> y          # addscalar 算子, 需要 op-plat 注册
```

op-plat 注册格式：

```json
// /op/op-cuda/add
{
  "symbols": ["+"],
  "dtype": ["f32", "f16", "bf16"],
  ...
}
```

VM 在解析中缀表达式时：
1. 检查操作数类型 → 是否全是标量？→ VM 直接求值
2. 操作数含张量 → 查 `/op/<backend>/<opcode>` 的 `symbols` 字段
3. 符号已注册 → 展开为对应 opcode，允许
4. 符号未注册 → 报错："op-plat 不支持 `+` 运算, 请使用显式前缀"

#### 为什么 `tensor.new` / `tensor.del` 而非 `alloc` / `free`

命名空间统一：

```
tensor.new    堆分配    — 与 opcode 名一致（heap-plat 的协议名）
tensor.del    堆释放    — 同上
tensor.clone  堆克隆    — 同上

matmul        计算      — op-plat 协议名
add           计算      — op-plat 协议名

栈变量无显式命名       — VM 遇到新变量名自动推导
```

好处：IR 中的名字就是 Redis 中的 opcode，零映射。

#### 为什么栈变量无需显式 new

```
# VM 执行:
sum(x) -> s          ← s 首次出现在写位置, VM 自动创建栈变量
s + 1.0 -> s         ← s 已存在, 复用

# 等价于 VM 内部:
# 第一次遇到 s: SET /vthread/<vtid>/s = {dtype: f32, value: ...}
# 后续遇到 s: GET /vthread/<vtid>/s → 读写
```

### 3.4 完整示例

```deepxir
# 线性函数 — 仅前缀调用
fn add_test(A: f32[?,?], B: f32[?,?]) -> C: f32[?,?] {
    matmul(A, B) -> C
}

# 带控制流 + 堆生命周期
fn dynamic_clamp(x: f32[?], min: f32, max: f32) -> y: f32[?] {
    tensor.new([?], f32) -> mask_low
    x < min -> mask_low                       # 中缀, 需 op-plat 注册 <

    if any(mask_low) {
        where(mask_low, min, x) -> y
    } else {
        y <- x                                # C风格
    }
    tensor.del(mask_low)

    tensor.new([?], f32) -> mask_high
    y > max -> mask_high                      # 中缀, 需 op-plat 注册 >

    if any(mask_high) {
        where(mask_high, max, y) -> y
    }
    tensor.del(mask_high)

    ret y
}

# 循环 + 混合前缀/中缀
fn training_step(data: f32[?,?]) -> loss: f32 {
    0.0 -> total                              # 标量直接用前缀

    for i in 0..100 {
        matmul(data, /models/W) -> pred       # 前缀, 堆引用
        pred - data -> err                    # 中缀, 需注册 -
        err * err -> sq                       # 中缀, 需注册 *
        total + sum(sq) -> total              # 中缀 + 前缀
    }
    total / 100.0 -> loss                     # 中缀, 需注册 /
    ret loss
}

# C风格写法（等价）
fn cstyle_example(x: f32[4]) -> y: f32[4] {
    s <- sum(x)                               # C风格前缀
    s > 0 -> cond                             # 中缀
    if cond {
        y <- x + 1.0                          # C风格中缀
    } else {
        y <- x * -1.0
    }
    ret y
}
```

### 3.5 中缀→前缀的展开规则（VM 解析层）

```
x + y -> z    →  add(x, y) -> z        (二元)
x - y -> z    →  sub(x, y) -> z
x * y -> z    →  mul(x, y) -> z
x / y -> z    →  div(x, y) -> z
x % y -> z    →  mod(x, y) -> z
x == y -> z   →  equal(x, y) -> z
x != y -> z   →  notequal(x, y) -> z
x < y -> z    →  less(x, y) -> z
x > y -> z    →  greater(x, y) -> z
x <= y -> z   →  lessequal(x, y) -> z
x >= y -> z   →  greaterequal(x, y) -> z
x && y -> z   →  and(x, y) -> z
x || y -> z   →  or(x, y) -> z
!x -> y       →  not(x) -> y           (单元)
-x -> y       →  neg(x) -> y           (单元)
```

每个目标 opcode 必须被对应后端注册了 `symbols` 字段才允许中缀形式。

### 3.6 Redis 存储格式（嵌套同构）

```
fn example(x: f32[4]) -> y: f32[4] {
    tensor.new([4], f32) -> a
    sum(x) -> s
    if s > 0 {
        x + 1.0 -> y
    } else {
        y <- x * -1.0
    }
    tensor.del(a)
    ret y
}

↓ Redis:
/src/func/example           = "fn example(x: f32[4]) -> y: f32[4]"
/src/func/example/0         = "tensor.new([4], f32) -> a"
/src/func/example/1         = "sum(x) -> s"
/src/func/example/2         = "if"
/src/func/example/2/cond    = "s > 0"
/src/func/example/2/then/0  = "x + 1.0 -> y"
/src/func/example/2/else/0  = "y <- x * -1.0"
/src/func/example/3         = "tensor.del(a)"
/src/func/example/4         = "ret y"
```

---

## 4. `->`/`<-` 与 SSA 对比分析

> **核心问题**：`->`/`<-` 方向语义和 SSA 单赋值模型，是变种等价关系还是根本不同的设计选择？

### 4.1 本质差异：两种不同的语义模型

```
┌─────────────────────────────────────────────────────────────────┐
│                     两种模型的核心分歧                            │
│                                                                 │
│  SSA  = 值身份模型 (Value Identity)                              │
│         追踪"哪个值"被使用                                       │
│         同一存储位置的不同时刻值 → 不同的 SSA 名字                │
│         问题域: 这个计算依赖哪个具体的值？                        │
│                                                                 │
│  ->/<- = 存储效应模型 (Storage Effect)                           │
│          追踪"哪个存储位置"被读写                                 │
│          同一名字的不同时刻值 → 同一个变量在不同时刻              │
│          问题域: 这个操作读写哪些 KV 路径？                       │
└─────────────────────────────────────────────────────────────────┘
```

**它们不是等价变种。** 两种模型回答的是不同层次的问题，服务于不同的系统角色。

### 4.2 形式化对比

| 维度 | SSA | `->`/`<-` |
|------|-----|----------|
| **变量赋值次数** | 严格 1 次 | 无限次（可变） |
| **值标识** | 虚拟寄存器编号 (`%0`, `%1`) | KV 路径 (`./x`, `/data/W`) |
| **数据流表达** | 操作数引用 = 隐式 use-def 链 | 显式 reads[] / writes[] 数组 |
| **控制流汇合** | block arguments (替代 φ 节点) | 变量自然复用，无需特殊机制 |
| **支配关系** | 严格支配树 (定义支配所有使用) | 无形式支配约束 |
| **副作用建模** | 困难 (需要特殊 dialect/op) | 天然 (tensor.new/del 就是指令) |
| **理论基础** | 编译器优化理论 (Cytron et al. 1991) | 分布式 KV 状态机 |
| **典型系统** | LLVM IR, MLIR, GCC GIMPLE | deepxir, Redis Lua, SQL stored procedures |

### 4.3 各自优势

#### SSA 优势

```
1. 编译器优化使能器
   %1 = add(%a, %b)        ← CSE: 同一操作数 → 同一结果, 可消除重复
   %2 = add(%a, %b)        ← 因为 %a, %b 的 SSA 值身份不变, CSE 直接判定 %1 == %2
   
   对比 ->/<-:
   add(a, b) -> c          ← CSE 困难: a, b 的值可能在指令间被改写 (可变变量)
   add(a, b) -> d          ← VM 无法仅从名字判定 c == d, 需要追踪 a, b 的最近写入

2. use-def 链零成本
   %result = mul(%x, %y)   ← %x 引用直接指向定义 %x 的唯一指令
   无需额外数据结构即可遍历数据流图

3. 寄存器分配友好
   SSA → 干涉图 → 着色 → 寄存器
   变量版本号天然就是活性区间标识

4. 形式化验证
   支配树 + 支配边界 → 可形式证明优化变换的正确性
   学术界 30+ 年理论积累
```

#### `->`/`<-` 优势

```
1. 执行协议直通
   add(a, b) -> c
   ↓ 零映射 ↓
   {"opcode": "add", "reads": ["/vthread/vt1/a", "/vthread/vt1/b"], "writes": ["/vthread/vt1/c"]}
   ↓ 直接发送到后端 ↓
   op-plat / heap-plat 执行

   对比 SSA: 需要额外 lowering 步骤
   %1 = add %0, %arg0  →  解析 use-def → 分配内存槽 → 生成 reads/writes

2. 分布式状态天然
   /vthread/vt1/a  ← KV 路径即变量身份
   跨节点执行: 同一路径在不同节点指向同一数据分片
   SSA 的值编号 (%0, %1) 是局部的, 无法跨节点引用

3. 副作用指令自然嵌入
   tensor.new([4], f32) -> ./a     ← 分配 GPU 显存, 这是一个副作用
   tensor.del(./a)                  ← 释放 GPU 显存, 这也是副作用
   
   SSA 中:
   %0 = tensor.new [4] : f32    ← MLIR 需要 memref dialect 专门处理
   tensor.del %0                 ← 破坏 SSA 纯函数假设, 需要特殊标注

4. 多写入直观
   split(x) -> (a, b)            ← 一目了然: 一个输入, 两个输出
   对比 SSA: 需要 tuple 封装再解构, 或自定义 dialect
```

### 4.4 同一示例的两种表达

```
问题: 计算 (x + y) * (x - y), 其中 x 是输入参数

┌─── SSA ──────────────────────┐  ┌─── ->/<- ────────────────────┐
│                               │  │                               │
│ fn calc(x: f32) -> f32 {      │  │ fn calc(x: f32) -> z: f32 {   │
│   %0 = add(x, y)              │  │   x + y -> sum                │
│   %1 = sub(x, y)              │  │   x - y -> diff               │
│   %2 = mul(%0, %1)            │  │   sum * diff -> z             │
│   ret %2                      │  │   ret z                       │
│ }                             │  │ }                             │
│                               │  │                               │
│ 特点:                         │  │ 特点:                         │
│ • %0, %1, %2 各出现 1 次      │  │ • sum 出现 2 次 (写, 读)     │
│ • 数据流: use-def 链隐式      │  │ • 数据流: -> 方向显式         │
│ • 适合做 CSE/复写传播          │  │ • 适合直接执行               │
└───────────────────────────────┘  └───────────────────────────────┘
```

这个简单例子中两者等价——SSA 的 `%0` 对应 `sum`, `%1` 对应 `diff`, `%2` 对应 `z`。
**差异在控制流和副作用场景下才显著。**

### 4.5 控制流场景下的关键差异

```
场景: if/else 分支后合并使用结果

┌─── SSA (MLIR 风格) ──────────────────────────────────────────┐
│                                                                │
│ fn branch_example(x: f32, flag: bool) -> f32 {                 │
│   br flag ? @then : @else                                      │
│                                                                │
│ @then:                                                         │
│   %t = add(x, 1.0)          ← 定义 %t                          │
│   br @merge(%t)             ← %t 作为 block argument 传递       │
│                                                                │
│ @else:                                                         │
│   %e = sub(x, 1.0)          ← 定义 %e                          │
│   br @merge(%e)             ← %e 作为 block argument 传递       │
│                                                                │
│ @merge(%y: f32):            ← %y 是 block argument             │
│   %r = mul(%y, 2.0)         ← 使用 %y (来自 %t 或 %e)          │
│   ret %r                                                       │
│ }                                                              │
│                                                                │
│ 机制: block argument 取代 φ 节点                                │
│ 优势: SSA 贯穿始终, %y 有唯一定义 (block parameter)             │
│ 代价: 需要传递 block argument, 前端/VM 需理解支配关系            │
└────────────────────────────────────────────────────────────────┘

┌─── ->/<- (deepxir 风格) ──────────────────────────────────────┐
│                                                                │
│ fn branch_example(x: f32, flag: bool) -> result: f32 {         │
│   if flag {                                                    │
│     x + 1.0 -> y             ← 写入 y (slot)                   │
│   } else {                                                     │
│     x - 1.0 -> y             ← 写入 y (同一个 slot)            │
│   }                                                            │
│   y * 2.0 -> result          ← 读取 y (无论哪个分支写的)       │
│   ret result                                                   │
│ }                                                              │
│                                                                │
│ 机制: 可变 slot 复用                                            │
│ 优势: 无需 block argument, 前端/VM 逻辑简单                     │
│ 代价: 无值身份追踪, y 被写两次 (违反 SSA), 优化器需额外分析      │
└────────────────────────────────────────────────────────────────┘
```

**核心洞见**: SSA 的 block argument 和 `->`/`<-` 的可变 slot 是**对偶设计**:
- SSA 用**参数化**解决多来源问题 (值从 block 参数传入)
- `->`/`<-` 用**存储复用**解决多来源问题 (同一个 slot 被不同路径写入)

### 4.6 副作用场景下的根本分歧

```
场景: 循环内分配和释放临时张量

┌─── SSA (MLIR 需 memref dialect) ──────────────────────────────┐
│                                                                │
│ %buf = memref.alloc()[4] : f32    ← 分配, 有副作用            │
│ scf.for %i = 0 to 10 {                                         │
│   %tmp = memref.alloc()[4] : f32  ← 每次迭代分配新 memref     │
│   linalg.fill %tmp, %i                                          │
│   linalg.add %buf, %tmp -> %buf                                 │
│   memref.dealloc %tmp              ← 释放                      │
│ }                                                              │
│                                                                │
│ 问题:                                                          │
│ • %buf 被循环内更新 → 不能用纯 SSA, 需 memref (可变内存抽象)    │
│ • memref 本质是"带副作用的内存槽" → SSA 的纯函数假设在此破裂   │
│ • MLIR 的解决: 用 memref dialect 隔离副作用, 保持其余 IR 纯 SSA │
└────────────────────────────────────────────────────────────────┘

┌─── ->/<- (deepxir 原生) ──────────────────────────────────────┐
│                                                                │
│ tensor.new([4], f32) -> buf          ← 堆分配, is_write=1     │
│ for i in 0..10 {                                               │
│   tensor.new([4], f32) -> tmp        ← 堆分配                 │
│   fill(tmp, i) -> tmp                ← 写入                   │
│   buf + tmp -> buf                   ← 读写 buf               │
│   tensor.del(tmp)                    ← 堆释放                 │
│ }                                                              │
│                                                                │
│ 优势:                                                          │
│ • 副作用 (new/del) 与计算 (add/fill) 在同一层次                │
│ • 每条指令天然携带 reads/writes → 执行层直接可用               │
│ • 无需 dialect 隔离 → 概念完整, 学习成本低                     │
│ • buf 被多次写入 → 可变 slot, 与 GPU 显存操作模型一致           │
└────────────────────────────────────────────────────────────────┘
```

**根本分歧点**: SSA 假设纯函数计算模型 (值不可变), 副作必须隔离到特定 dialect。
`->`/`<-` 假设可变状态模型 (存储位置可更新), 副作用是第一公民。

deepxir 的核心场景——GPU 显存分配/释放、分布式 KV 读写、多后端异构执行——**本质上就是副作用密集的**。
因此 `->`/`<-` 是更自然的匹配。

### 4.7 等价性证明

**在纯计算 (无副作用、无控制流汇合) 的线性代码段内, SSA 与 `->`/`<-` 严格等价**:

```
定理 1 (线性段等价):
  对于不含控制流分支和副作用指令的线性指令序列,
  SSA 表示和 ->/<- 表示之间存在双射 (bijection)。

  证明:
  SSA:  v0 = op(args...)      →/<-:       op(args...) -> name
  映射: 每个 SSA 虚拟寄存器 %i  ←→  首次出现在写位置的变量名 name
        每个 SSA use           ←→  读位置的变量名
  映射是双射: 每个 %i 仅有一个定义点, 每个定义点产生一个新 %i,
              ->/<- 侧每个写位置引入一个变量名, 与 SSA 一一对应。
```

**在控制流汇合场景下, 语义等价但结构不等价**:

```
定理 2 (控制流汇合等价):
  对于带 φ 节点的 SSA 形式,
  存在一个 ->/<- 程序与之计算等价 (compute the same results),
  但结构上不等价 (φ 通过 block parameter 或 slot 复用实现)。

  映射方向:
  SSA φ:  %y = φ(%t, %e)    →  ->/<-:  在两个分支中写入同一个 slot y
  
  逆映射方向:
  ->/<-:  分支中写 slot y   →  SSA:     插入 φ 节点或 block argument
  
  关键差异:
  SSA → ->/<- 是信息丢失的 (值身份信息丢失, 只保留了存储位置信息)
  ->/<- → SSA 是信息恢复的 (需要做 SSA 构造算法, 为标准编译器技术)
```

**在副作用场景下, 两者不等价**:

```
定理 3 (副作用分歧):
  对于包含内存分配/释放、I/O 等副作用的程序,
  SSA (纯 MLIR core) 与 ->/<- 不等价。

  原因:
  SSA 将副作用隔离到特定 dialect (memref, gpu, async),
  副作用指令的语义由 dialect 定义, 与纯 SSA 值流分离。

  ->/<- 将副作用嵌入每条指令的 reads/writes 数组中,
  副作用与数据流在同一表示中融合。

  两种表示的计算结果等价, 但抽象模型不同:
  SSA 分离了"值计算"和"效应执行"两个世界
  ->/<- 将世界统一为"对 KV 空间的读写操作"
```

### 4.8 互补共存：C2 架构中的双重表示

C2 架构的两层 IR 正是利用了两种模型的互补性：

```
┌────────────────────────────────────────────────────────────────┐
│                      C2 双层 IR 格局                           │
│                                                                │
│  Layer 1: 结构化 IR (前端输出, ->/<-)                          │
│  ┌──────────────────────────────────────────────────┐        │
│  │ fn example(x: f32[4]) -> y: f32[4] {              │        │
│  │   tensor.new([4], f32) -> a                      │        │
│  │   sum(x) -> s                                     │        │
│  │   if s > 0 {               ← 结构化控制流         │        │
│  │     x + 1.0 -> y                                 │        │
│  │   } else {                                        │        │
│  │     y <- x * -1.0          ← C风格箭头            │        │
│  │   }                                               │        │
│  │   tensor.del(a)                                  │        │
│  │   ret y                                           │        │
│  │ }                                                 │        │
│  └──────────────────────────────────────────────────┘        │
│         │                                                     │
│         │  Scheduler Lowering                                │
│         ▼                                                     │
│  Layer 2: 基本块 IR (VM 执行, SSA 可选)                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ @0:                                               │        │
│  │   %0 = tensor.new [4] : f32     ← 块内 SSA (可选) │        │
│  │   %1 = sum %x                                     │        │
│  │   %2 = greater %1, 0                             │        │
│  │   br %2 ? @1 : @2              ← 平铺 CFG         │        │
│  │                                                    │        │
│  │ @1:                                                │        │
│  │   %3 = add %x, 1.0                               │        │
│  │   jump @3                                          │        │
│  │                                                    │        │
│  │ @2:                                                │        │
│  │   %4 = mul %x, -1.0                              │        │
│  │   jump @3                                          │        │
│  │                                                    │        │
│  │ @3(%y: f32):                ← block argument       │        │
│  │   tensor.del %0                                  │        │
│  │   ret %y                                           │        │
│  └──────────────────────────────────────────────────┘        │
│                                                                │
│  关键: 两种模型各司其职                                         │
│  • Layer 1 (->/<-): 面向人、Agent、前端 — 可读、可生成         │
│  • Layer 2 (SSA+CFG): 面向 VM、优化器 — 可分析、可变换         │
│  • Lowering 是有损变换: 高层语义展开, 值身份重构                │
└────────────────────────────────────────────────────────────────┘
```

### 4.9 为什么 deepxir 选择 `->`/`<-` 作为主语法而非 SSA

| 决策因素 | `->`/`<-` | SSA | 分析 |
|---------|-----------|-----|------|
| **协议对齐** | 直接映射到 reads[]/writes[] | 需要额外 lowering | deepxir 的协议层就是以 reads/writes 为单位的, 强行用 SSA 增加无谓转换 |
| **前端生成复杂度** | 低 (变量名字直接映射) | 高 (需要 SSA 构造算法, 处理 φ/block arg) | Python/Go 前端生成 `->`/`<-` 只需记录变量名; 生成 SSA 需维护版本计数器 |
| **人可读性** | 高 (名字有意义: `./loss`, `./grad`) | 中 (编号 `%0`, `%1` 无意义, 需注释) | 调试时 `./loss` 比 `%42` 更直观 |
| **分布式语义** | 天然 (路径即全局标识) | 需额外映射 (值编号是局部的) | deepxir 是分布式系统, KV 路径就是全局标识 |
| **副作用表达** | 自然 (new/del 就是指令) | 需要 dialect 隔离 | 不需要引入 dialect 概念, 降低系统复杂度 |
| **优化能力** | 需额外分析 | 天然支持 | 但 deepxir 当前优化需求不高 (计算在 GPU 上), 且 C2 Scheduler 内部可以重建 SSA |

**结论: `->`/`<-` 是 deepxir 的正确主语法选择。SSA 作为 C2 Scheduler 内部的优化表示, 在 lowering 阶段构建, 对前端和 VM 透明。**

### 4.10 决策总结

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   SSA 和 ->/<- 不是等价变种。                                  │
│                                                              │
│   它们是两种不同的语义模型:                                     │
│   • SSA 建模值身份 (适合编译器分析和优化)                      │
│   • ->/<- 建模存储效应 (适合分布式执行和协议直通)              │
│                                                              │
│   对 deepxir 而言:                                            │
│   • 主语法用 ->/<- — 面向人、Agent、前端、协议                │
│   • SSA 作为 C2 Scheduler 内部优化 IR — 对用户透明             │
│   • 两者通过 Scheduler lowering 连接                          │
│   • C1 模式不引入 SSA (保持单层简单)                          │
│   • C2 模式在 lowering 时重建 SSA (使能优化)                  │
│                                                              │
│   这不是非此即彼的选择, 而是分层的职责分配。                    │
│   MLIR 的选择也佐证了这一点: scf (结构化) = 类比 ->/<-,        │
│   cf (底层) = SSA + 基本块, 两层并存。                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 方案 C1：单层关键字 IR

前后端共用 deepxir 语法。VM 直接解释控制流关键字，管理嵌套作用域。

### 5.1 VM 执行模型

```go
func Execute(vtid string) {
    inst := decode(currentPC)
    switch inst.Opcode {
    case "tensor.new":
        pushToHeapPlat(vtid, inst)       // → heap-plat
    case "tensor.del":
        pushToHeapPlat(vtid, inst)
    case "if":
        cond := evalOperand(inst.Cond)
        if isTruthy(cond) {
            pushScope(vtid, inst.Then)
        } else {
            pushScope(vtid, inst.Else)
        }
    case "for":
        pushLoopScope(vtid, inst.Var, inst.Start, inst.End, inst.Body)
    case "while":
        pushWhileScope(vtid, inst.Cond, inst.Body)
    case "break":
        popLoopScope(vtid)
    case "continue":
        rewindToLoopCond(vtid)
    case "ret":
        popFrame(vtid)
    default:
        dispatch(inst)  // 计算 → op-plat / VM 求值
    }
}
```

### 5.2 架构评估

| 准则 | 评分 |
|------|------|
| 概念完整性 | ⭐⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐ |

---

## 6. 方案 C2：二层 IR + Scheduler Lowering

```
前端 → deepxir (C1 语法)
         │
    [Scheduler]
    Pass 1: Region Flattening  →  基本块
    Pass 2: 块内 SSA 构造
    Pass 3: 优化
    Pass 4: Target Emission
         │
         ▼
    平铺基本块 → VM
```

基本块 IR 格式（C2 Scheduler 输出，VM 输入）：

```
fn example(x: f32[4]) -> y: f32[4] {
@0:
    tensor.new([4], f32) -> a
    sum(x) -> s
    greater(s, 0) -> cond          # 展开为前缀, 不做中缀
    br cond ? @1 : @2

@1:
    add(x, 1.0) -> y               # 全部前缀, 无中缀
    jump @3

@2:
    mul(x, -1.0) -> y
    jump @3

@3:
    tensor.del(a)
    ret y
}
```

每个 block 有恰好一条终止指令：`br cond ? @t : @f` | `jump @target` | `ret [val]`。
中缀在 lowering 时全部展开为前缀 opcode。

### 架构评估

| 准则 | 评分 |
|------|------|
| 关注点分离 | ⭐⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐⭐ |
| 概念完整性 | ⭐⭐⭐⭐⭐ |
| **综合** | **25/25** |

---

## 7. 方案 C3/C4/C5 精简

| 方案 | 核心 | 亮点 | 架构评分 |
|------|------|------|---------|
| C3 单层基本块 | 前端直接输出基本块（C2 VM 输入格式） | VM 极简 | 19/25 |
| C4 单层 Region | VM 原生理解嵌套 region | MLIR 概念对齐 | 22/25 |
| C5 关键字+VM内lower | 前端输出 C1，VM 内部 lowering | 对外简单对内优化 | 19/25 |

---

## 8. 五方案横向对比

| 准则 | C1 关键字 | C2 二层+Scheduler | C3 单层块 | C4 Region | C5 关键字+VM内 |
|------|---------|------------------|----------|----------|------------|
| 关注点分离 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 概念完整性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **综合** | **19** | **25** | **19** | **22** | **19** |

### 关键维度对比

```
前端负担:  C3 > C1 = C4 = C5 > C2
VM 复杂度: C1 = C4 > C5 > C2 = C3
运维复杂度: C2 > C1 = C3 = C4 = C5
优化能力:  C2 > C5 > C1 = C3 = C4
```

---

## 9. 结论

**架构最优：C2（二层 IR + Scheduler）** — 25/25，所有维度满分。

**C1 与 C2 互补**：
1. **先 C1**：deepxir 语法 → VM 直接解释，调试友好，快速可用
2. **再 C2**：加 Scheduler 做 lowering + 优化，C1 语法即 C2 的结构化 IR 层格式
3. VM 执行层不变（基本块格式），Scheduler 可插拔

**三个核心约束贯穿所有方案**：
- `->` / `<-` 保留方向语义，不可用 `=` 替代
- `tensor.new` / `tensor.del` 统一命名，栈变量 VM 自推导
- 中缀运算符需 op-plat 注册 `symbols` 字段
