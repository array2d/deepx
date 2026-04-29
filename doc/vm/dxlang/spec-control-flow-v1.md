# DeepX 控制流与前端代码生成 — 架构分析 v1

> 方案 C（二层 IR）确定为最优架构方向。本文基于 C 的核心思想——关注点分离、
> 语义等价的格式变换——展开 5 种子方案。

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

"几层 IR"和"lowering 在哪里"是实现细节：

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

> 语法必须同时满足三者的阅读需求：**人**（直观、可扫读）、**VM**（确定可解析）、**Agent**（结构化可生成）。

### 3.1 设计原则

| 原则 | 含义 | 体现 |
|------|------|------|
| **数据流先行** | 结果在左，操作在右 | `y = add(x, 1.0)` 而非 `add(x, 1.0) -> y` |
| **最少标点** | 去掉无信息量的符号 | `x` 而非 `./x`，`f32` 而非 `tensor<f32>` |
| **视觉层次** | 嵌套通过缩进+大括号表达 | 控制流体的层级一目了然 |
| **关键字优于符号** | 文字描述意图，符号留给算术 | `alloc`/`free` 而非 `newtensor`/`deltensor` |
| **一种赋值方式** | 全用 `=`，消除 `->`/`<-` 混淆 | 统一左值 = 右值 |

### 3.2 完整语法

```
program      = func*
func         = "fn" name "(" params ")" ("->" returns)? body
params       = param ("," param)*
param        = name ":" type
returns      = name ":" type  |  "(" name ":" type ("," name ":" type)* ")"
type         = "f16" | "f32" | "f64" | "bf16" | "i8" | "i16" | "i32" | "i64" | "bool" | "string"
               | type "[" dims "]"            # 固定形状: f32[2,4]
               | type "[" "?" ("," "?")* "]"  # 动态形状: f32[?,?]
body         = "{" stmt* "}"
stmt         = assign | ctrl_if | ctrl_for | ctrl_while | ctrl_break | ctrl_continue
               | ctrl_switch | ctrl_return | lifecycle | bare_expr
assign       = name "=" expr
bare_expr    = expr                           # 无返回值操作 (如 free x)
lifecycle    = "alloc" type              # 分配张量
             | "free" name                   # 释放张量
expr         = infix | prefix | literal | path
infix        = expr binop expr               # a + b, s > 0
prefix       = unop expr                     # !flag, -x
             | name "(" args ")"             # matmul(a, b), relu(x)
binop        = "+" | "-" | "*" | "/" | "%"
             | "==" | "!=" | "<" | ">" | "<=" | ">="
             | "&&" | "||"
unop         = "-" | "!"
args         = expr ("," expr)*
literal      = int | float | "true" | "false" | string
path         = name                          # 局部变量: x, tmp
             | "/" name ("/" name)*          # 堆路径: /models/W
ctrl_if      = "if" expr body ("else" (ctrl_if | body))?
ctrl_for     = "for" name "in" range body
range        = expr ".." expr                # 0..10
ctrl_while   = "while" expr body
ctrl_loop    = "loop" body                   # 无限循环
ctrl_break   = "break"
ctrl_continue = "continue"
ctrl_switch  = "switch" expr "{" case* default? "}"
case         = "case" literal ":" body
default      = "default" ":" body
ctrl_return  = "ret" expr?
```

### 3.3 与旧语法的逐项对比

```
旧语法                                    新语法                       改进点
─────────────────────────────────────    ─────────────────────────    ──────
def add_test(A:tensor, B:tensor)         fn add_test(A: f32, B: f32)  去掉 tensor<>，类型即类型
  -> (C:tensor) {                          -> C: f32 {

newtensor("f32", "[4]") -> ./a           a = alloc f32[4]             左值在前，去掉引号
deltensor(./a)                           free a                       关键字，不是函数调用

add(./x, 1.0) -> ./y                     y = x + 1.0                  中缀算术，去掉 ./
matmul(./A, ./W) -> ./y                  y = matmul(A, W)             前缀调用，去掉 ./

if (greater(./s, 0)) {                   if s > 0 {                   条件中缀，去掉括号套函数
    add(./x, 1.0) -> ./y                     y = x + 1.0
} else {                                 } else {
    mul(./x, -1.0) -> ./y                    y = x * -1.0
}                                        }

for (var i=0; less(i,10);                for i in 0..10 {              range 语法，清晰直观
      add(i,1) -> i) {                       y = y + i
    add(./y, i) -> ./y                   }
}

while (greater(./y, 100.0)) {            while y > 100.0 {            条件中缀
    mul(./y, 0.5) -> ./y                     y = y * 0.5
}                                        }

return(./y)                              ret y                        关键字，去掉括号
and(./a, ./b) -> ./c                     c = a && b                   逻辑运算符中缀
```

### 3.4 完整示例

```deepxir
# 线性函数
fn add_test(A: f32[?,?], B: f32[?,?]) -> C: f32[?,?] {
    C = matmul(A, B)
}

# 带控制流
fn dynamic_clamp(x: f32[?], min: f32, max: f32) -> y: f32[?] {
    mask_low = x < min
    if any(mask_low) {
        y = where(mask_low, min, x)
    } else {
        y = x
    }

    mask_high = y > max
    if any(mask_high) {
        y = where(mask_high, max, y)
    }

    ret y
}

# 循环 + 生命周期
fn fused_linear(A: f32[?,?], W: f32[?,?], b: f32[?]) -> out: f32[?,?] {
    mm = alloc f32[?,?]
    mm = matmul(A, W)

    bias = alloc f32[?,?]
    bias = add(mm, b)
    free mm

    out = relu(bias)
    free bias
}

# 多层循环 + 堆引用
fn training_step(model_W: f32[?,?], data: f32[?,?]) -> loss: f32 {
    total = 0.0
    for i in 0..100 {
        pred = matmul(data, /models/W)
        err = pred - data
        sq = err * err
        total = total + sum(sq)
    }
    loss = total / 100.0
    ret loss
}
```

### 3.5 为什么这个语法是"可读的"

| 读者 | 旧语法的障碍 | 新语法如何解决 |
|------|-----------|------------|
| **人** | `./` 前缀刺眼，`@0:` 像乱码，`->` 和 `<-` 混用困惑 | 像写伪代码：`y = x + 1`、`if s > 0`、`for i in 0..10` |
| **VM** | 需处理两种箭头、多种括号、引号类型字符串 | 统一 `=` 赋值，纯中缀或纯前缀，固定关键字，确定性解析 |
| **Agent** | 引号嵌套易错，`./` 需额外生成逻辑 | 没有引号包裹的类型，变量名即变量名，模式固定可预测 |

### 3.6 中缀运算符到 opcode 的映射（VM 解析层）

中缀是语法糖。VM 解析时还原为 opcode：

```
y = x + 1.0    →  add(x, 1.0) -> y
y = x - 1.0    →  sub(x, 1.0) -> y
y = x * 1.0    →  mul(x, 1.0) -> y
c = a > 0      →  greater(a, 0) -> c
c = a && b     →  and(a, b) -> c
c = !a         →  not(a) -> c
```

这层映射是确定的、可逆的、零歧义的。前端可以输出中缀或前缀，VM 看到的都是 opcode。

---

## 4. 方案 C1：单层关键字 IR

### 4.1 设计

**前后端共用 deepxir 语法。** VM 直接解释控制流关键字，管理嵌套作用域。

```
前端 → deepxir 文本 (写入 Redis /src/func/<name>)
         │
    VM 加载 → 解析 → 逐条执行
         │
    if/for/while → pushScope 进入子作用域
    break/continue → popScope 退出作用域
```

### 4.2 Redis 存储格式

IR 嵌套结构与 Redis key 嵌套同构：

```
fn example(x: f32[4]) -> y: f32[4] {
    a = alloc f32[4]
    s = sum(x)
    if s > 0 {
        y = x + 1.0
    } else {
        y = x * -1.0
    }
    free a
    ret y
}

↓ Redis:
/src/func/example           = "fn example(x: f32[4]) -> y: f32[4]"
/src/func/example/0         = "a = alloc f32[4]"
/src/func/example/1         = "s = sum(x)"
/src/func/example/2         = "if"
/src/func/example/2/cond    = "s > 0"
/src/func/example/2/then/0  = "y = x + 1.0"
/src/func/example/2/else/0  = "y = x * -1.0"
/src/func/example/3         = "free a"
/src/func/example/4         = "ret y"
```

### 4.3 VM 执行模型

```go
func Execute(vtid string) {
    inst := decode(currentPC)
    switch inst.Opcode {
    case "if":
        cond := evalExpr(inst.Cond)
        if cond {
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
        dispatch(inst)  // 计算/生命周期 → op-plat / heap-plat / VM 求值
    }
}
```

### 4.4 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| 关注点分离 | ⭐⭐⭐ | 前后端共用一套 IR（好），VM 需理解嵌套语义（偏重） |
| 单一抽象级别 | ⭐⭐⭐⭐ | 关键字和计算指令在同一层 |
| 可验证性 | ⭐⭐⭐ | 关键字嵌套可 schema 校验 |
| 可扩展性 | ⭐⭐⭐⭐ | 新增关键字只需 VM handler |
| 概念完整性 | ⭐⭐⭐⭐⭐ | "指令流中嵌入控制流关键字，嵌套同构 Redis key" |

---

## 5. 方案 C2：二层 IR + Scheduler Lowering

### 5.1 设计

```
前端 → deepxir (C1 语法)
         │
    [Scheduler]
    Pass 1: Region Flattening  →  基本块 (@0/@1)
    Pass 2: 块内 SSA 构造
    Pass 3: 优化
    Pass 4: Target Emission
         │
         ▼
    平铺基本块 → VM
```

### 5.2 基本块 IR 语法（Scheduler 输出，VM 输入）

```
# C2 基本块格式 — 干净且可验证
fn example(x: f32[4]) -> y: f32[4] {
@0:
    a = alloc f32[4]
    s = sum(x)
    cond = s > 0
    br cond ? @1 : @2

@1:
    y = x + 1.0
    jump @3

@2:
    y = x * -1.0
    jump @3

@3:
    free a
    ret y
}
```

每个 block 有恰好一条终止指令：`br cond ? @t : @f` 或 `jump @target` 或 `ret [val]`。

### 5.3 架构评估

| 准则 | 评分 |
|------|------|
| 关注点分离 | ⭐⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐⭐ |
| 概念完整性 | ⭐⭐⭐⭐⭐ |
| **综合** | **25/25** |

---

## 6. 方案 C3：单层基本块 IR

前端直接生成基本块格式（与 C2 的 Scheduler 输出相同），无中间服务。

| 准则 | 评分 |
|------|------|
| 单一抽象级别 | ⭐⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐⭐ |
| 关注点分离 | ⭐⭐⭐ |

---

## 7. 方案 C4：单层 Region IR

VM 原生理解 Region（嵌套作用域）。Region 是通用抽象，`if`/`for`/`while` 是其类型。

```
fn example(x) -> y {
    region {
        a = alloc f32[4]
        s = sum(x)
        region.if s > 0 {
            y = x + 1.0
        } region.else {
            y = x * -1.0
        }
        free a
        ret y
    }
}
```

| 准则 | 评分 |
|------|------|
| 概念完整性 | ⭐⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐⭐⭐ |

---

## 8. 方案 C5：关键字 IR + VM 内 Lowering

前端输出 C1 deepxir。VM 加载时内部 lowering 为基本块。对外一层，对内两层。

| 准则 | 评分 |
|------|------|
| 可验证性 | ⭐⭐⭐⭐ |
| 关注点分离 | ⭐⭐⭐⭐ |

---

## 9. 五方案横向对比

| 准则 | C1 关键字 | C2 二层+Scheduler | C3 单层块 | C4 Region | C5 关键字+VM内 |
|------|---------|------------------|----------|----------|------------|
| 关注点分离 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 概念完整性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **综合** | **19** | **25** | **19** | **22** | **19** |

### 关键区分维度

```
IR 复杂度 (前端负担):
  C3 > C1 = C4 = C5 > C2

VM 复杂度 (执行负担):
  C1 = C4 > C5 > C2 = C3

运维复杂度:
  C2 > C1 = C3 = C4 = C5

优化能力:
  C2 > C5 > C1 = C3 = C4
```

---

## 10. 结论

**架构最优：C2（二层 IR + Scheduler）** — 25/25，所有维度满分。

**C1（单层关键字 IR）在概念完整性上同样满分。** C1 和 C2 **互补而非互斥**：

1. **先 C1**：VM 直接解释 deepxir 关键字，快速可用，调试友好
2. **再 C2**：加 Scheduler 做 lowering + 全局优化，VM 执行层不变
3. C1 文本格式作为 C2 的**结构化 IR 层格式**——前端和 Scheduler 在同一语法上协作
