# DeepX 控制流与前端代码生成 — 架构分析 v1

> 方案 C（二层 IR）确定为最优架构方向。本文基于 C 的核心思想——关注点分离、
> 语义等价的格式变换——展开 4 种子方案，分别对应不同的 IR 层数和 lowering 位置。

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

方案 C 的核心思想不是"要有两层 IR"，而是：

> **控制流从"用户意图"到"机器执行"是一个语义等价的格式变换过程。
> 这个变换应该发生在明确的边界上，每步变换可独立验证。**

"几层 IR"和"lowering 在哪里"是这个思想的**实现细节**。不同的子方案在这个轴上有不同选择：

```
                    IR 层数          Lowering 位置
C1 单层关键字        1 层            无（VM 直接解释）
C2 二层分离          2 层            Scheduler 服务
C3 单层基本块        1 层            前端负责
C4 单层 Region       1 层            VM 原生执行
C5 关键字+VM内lower  1 层（外部）    VM 加载时内部 lowering
                    2 层（内部）
```

---

## 3. 方案 C1：单层关键字 IR（用户提议）

### 3.1 设计

**前后端共用同一套 IR。** 控制流以关键字形式嵌入指令流，VM 直接解释执行，
不做 lowering 到基本块。

```
IR 格式 (文本或 JSON)：

def example(x: tensor<f32>[4]) -> (y: tensor<f32>[4]) {
    newtensor("f32", "[4]") -> ./a
    sum(./x) -> ./s
    
    if (greater(./s, 0)) {              ← 关键字，不是 br
        add(./x, 1.0) -> ./y
    } else {
        mul(./x, -1.0) -> ./y
    }
    
    for (var i = 0; less(i, 10); add(i, 1) -> i) {
        add(./y, i) -> ./y
    }
    
    while (greater(./y, 100.0)) {
        mul(./y, 0.5) -> ./y
    }
    
    deltensor(./a)
    return(./y)
}
```

**控制流关键字**：

| 关键字 | 语义 | VM 行为 |
|--------|------|--------|
| `if (cond) { ... } [else { ... }]` | 条件分支 | 求值 cond → 进入 then 或 else 子作用域 |
| `for (init; cond; step) { ... }` | 计数循环 | 执行 init → 循环: 求值 cond → 执行 body → 执行 step |
| `while (cond) { ... }` | 条件循环 | 循环: 求值 cond → 执行 body |
| `loop { ... }` | 无限循环 | 循环 body，直到内部 break |
| `break` | 跳出最近循环 | 跳出当前循环作用域 |
| `continue` | 跳过本次迭代 | 跳到循环条件判断 |
| `switch (val) { case v1: ... case v2: ... default: ... }` | 多路分支 | 匹配 val 到 case |

### 3.2 VM 执行模型

```go
// VM 直接解释控制流关键字，管理嵌套作用域
func Execute(vtid) {
    inst := decode(pc)
    
    switch inst.Opcode {
    case "if":
        cond := eval(inst.Cond)
        if cond {
            pushScope(vtid, inst.ThenBody)  // 进入 then 子作用域
        } else if inst.ElseBody != nil {
            pushScope(vtid, inst.ElseBody)
        }
        
    case "for":
        eval(inst.Init)
        pushLoopScope(vtid, inst.Cond, inst.Step, inst.Body)
        
    case "while":
        pushLoopScope(vtid, inst.Cond, nil, inst.Body)
        
    case "break":
        popLoopScope(vtid)
        
    case "continue":
        jumpToLoopCond(vtid)
        
    case "return":
        popFrame(vtid)
        
    default:
        // 计算/生命周期指令 → op-plat / heap-plat / VM 求值
        dispatch(inst)
    }
}
```

### 3.3 Redis 存储格式

控制流关键字在 Redis 中以嵌套 key 表达：

```
/vthread/<vtid>/[0,0]       = "newtensor"
/vthread/<vtid>/[0,0]/0,1   = "/data/a"

/vthread/<vtid>/[1,0]       = "if"
/vthread/<vtid>/[1,0]/cond  = "./cond"        ← 条件引用
/vthread/<vtid>/[1,0]/then/0     = "add"      ← then 子树
/vthread/<vtid>/[1,0]/then/0,-1  = "./x"
/vthread/<vtid>/[1,0]/else/0     = "mul"      ← else 子树
/vthread/<vtid>/[1,0]/else/0,-1  = "./x"

/vthread/<vtid>/[2,0]       = "for"
/vthread/<vtid>/[2,0]/init/0 = "var"
/vthread/<vtid>/[2,0]/cond   = "less(./i, 10)"
/vthread/<vtid>/[2,0]/step/0 = "add(./i, 1)"
/vthread/<vtid>/[2,0]/body/0 = "add(./y, ./i)"
```

nesting 自然表达了控制流的层次结构。

### 3.4 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| **关注点分离** | ⭐⭐⭐ | 前后端共用一套 IR（好），但 VM 需要理解控制流嵌套语义（VM 职责偏重） |
| **单一抽象级别** | ⭐⭐⭐⭐ | 关键字和计算指令在同一层，抽象级别一致 |
| **可验证性** | ⭐⭐⭐ | 关键字嵌套可 schema 校验（if 必须有 cond），但无法做基本块级别的跳转合法性验证 |
| **可扩展性** | ⭐⭐⭐⭐ | 新增控制流关键字只需 VM 加 handler，不影响前端 IR 格式 |
| **概念完整性** | ⭐⭐⭐⭐⭐ | 核心思想清晰："指令流中嵌入控制流关键字，VM 解释关键字管理嵌套作用域" |

**核心优势**：
- 无需 lowering 步骤，无 Scheduler 服务
- IR 结构和 Redis 存储结构完全同构（嵌套 key 对应嵌套语义）
- break/continue 等循环控制关键字直接可用
- VM 的执行逻辑与 IR 结构直观对应，调试友好

**架构代价**：
- VM 需要管理嵌套作用域栈（scope stack），增加了 VM 复杂度
- 无法做基本块级别的全局优化（因为没有 flatten 步骤）
- 循环展开、分支合并等优化需要 VM 在执行期自行判断

---

## 4. 方案 C2：二层 IR + Scheduler Lowering

**这是原方案 C。**

### 4.1 设计

```
前端 → 结构化 IR (JSON, if/for/while as regions)
         │
    [Scheduler]
    Pass 1: Region Flattening → 基本块
    Pass 2: SSA 构造 (块内)
    Pass 3: 优化
    Pass 4: Target Emission
         │
         ▼
    平铺基本块 IR → VM
```

### 4.2 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| **关注点分离** | ⭐⭐⭐⭐⭐ | 三层独立：前端→结构化 / Scheduler→变换 / VM→执行 |
| **单一抽象级别** | ⭐⭐⭐⭐⭐ | 每层一种格式，无混杂 |
| **可验证性** | ⭐⭐⭐⭐⭐ | 每层边界独立验证；块终止指令的合法性可形式化检查 |
| **可扩展性** | ⭐⭐⭐⭐⭐ | 新优化 pass 独立添加，不触及其他层 |
| **概念完整性** | ⭐⭐⭐⭐⭐ | lowering 链：结构化意图 → 语义等价变换 → 基本块执行 |

**核心优势**：
- 允许全局优化（跨块的死代码消除、块合并、循环不变量外提）
- VM 极简（只理解 br/jump/return）
- 前端和后端完全解耦

**架构代价**：
- 引入 Scheduler 服务，增加运维复杂度

---

## 5. 方案 C3：单层基本块 IR

### 5.1 设计

**前端直接生成基本块。** 没有结构化 IR 层，没有 Scheduler。VM 看到的和前端生成的完全一致。

```
前端负责：
  1. 将用户的 if/for/while 展开为基本块
  2. 分配 block ID (@0, @1, ...)
  3. 插入 br/jump 终止指令
  4. 建立 block 之间的跳转图

前端输出 (也是 VM 输入)：
  @0:
    newtensor(...) -> ./a
    br ./cond, @1, @2
  @1:
    add(./x, 1.0) -> ./y
    jump @3
  @2:
    mul(./x, -1.0) -> ./y
    jump @3
  @3:
    deltensor(./a)
    return
```

### 5.2 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| **关注点分离** | ⭐⭐⭐ | 前端做了 lowering 工作（结构→块），前端职责比 C1/C2 重 |
| **单一抽象级别** | ⭐⭐⭐⭐⭐ | 仅一种 IR 格式（基本块），概念统一 |
| **可验证性** | ⭐⭐⭐⭐ | 块格式天然可验证（每个块必须有终止指令），比 C1 强 |
| **可扩展性** | ⭐⭐⭐ | 新增控制流语义需要前端更新 lowering 逻辑 |
| **概念完整性** | ⭐⭐⭐⭐ | 核心思想："前端输出即 VM 输入，格式是最小执行语义" |

**核心优势**：
- VM 极简（与 C2 相同）
- 单一格式贯穿，无转换损耗
- 无中间服务

**架构代价**：
- 每个前端（Python/Go/C++）必须各自实现结构→块 lowering，逻辑重复
- 无法做全局优化（各前端 lowering 质量不一致）

---

## 6. 方案 C4：单层 Region IR

### 6.1 设计

**前后端共用 Region IR。** Region 是带作用域隔离的嵌套指令块。与 C1（关键字）的区别在于：C1 的控制流关键字是特殊的 opcode，C4 则将嵌套结构本身作为 IR 的核心抽象。

```
Region IR:
  Region 是一个有独立变量作用域、可包含子 Region 的指令容器。

def example(x) -> (y) {
    @entry:                               ← 一个 region
        newtensor("f32", "[4]") -> ./a
        sum(./x) -> ./s
        greater(./s, 0) -> ./cond
        region.if ./cond {                ← 子 region: then
            add(./x, 1.0) -> ./y
        } region.else {                   ← 子 region: else
            mul(./x, -1.0) -> ./y
        }
        
        region.while greater(./y, 100.0) { ← 子 region: loop body
            mul(./y, 0.5) -> ./y
        }
        
        deltensor(./a)
        return(./y)
}
```

**VM 执行**：维护一个 region 栈，进入子 region 时 push，离开时 pop。区域内的变量在 pop 时清理。

### 6.2 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| **关注点分离** | ⭐⭐⭐⭐ | Region 天然隔离作用域，比 C1 的关键字更结构化 |
| **单一抽象级别** | ⭐⭐⭐⭐ | Region 是唯一的组合原语 |
| **可验证性** | ⭐⭐⭐⭐⭐ | Region 嵌套合法性、变量作用域可形式化验证 |
| **可扩展性** | ⭐⭐⭐⭐ | 新 region 类型（如 region.parallel）直接添加 |
| **概念完整性** | ⭐⭐⭐⭐⭐ | 核心："Region = 作用域隔离"，与 MLIR 的 region 概念一致 |

**vs C1（关键字 IR）**：
- C1 把 if/for 当作特殊的 opcode，每条控制流有独立 handler
- C4 把 region 作为通用抽象，if/for/while 是 region 的不同实例化方式
- C4 更统一（一种机制解决多种控制流），C1 更直观（每种控制流有专属语法）

---

## 7. 方案 C5：关键字 IR + VM 内 Lowering

### 7.1 设计

**前端输出 C1 关键字 IR。** VM 加载时内部将关键字 IR lowering 为基本块。
没有独立的 Scheduler 服务，lowering 发生在 VM 进程内。

```
前端 → C1 关键字 IR (写入 Redis /src/func/<name>)
         │
    VM 加载时:
    1. 读取关键字 IR
    2. translate.go 新增 Lowering pass:
       if/for/while → 展开为 @0/@1/@2 基本块 + br/jump
    3. 写入 /vthread/<vtid>/ (执行层)
    4. 执行循环使用 block-loop (与 C2/C3 相同)
```

**关键**：外部看只有一层 IR（关键字），VM 内部有两层（关键字 → 块）。
lowering 是 VM 的一个模块，不是独立服务。

### 7.2 架构评估

| 准则 | 评分 | 分析 |
|------|------|------|
| **关注点分离** | ⭐⭐⭐⭐ | 外部：前端只管生成关键字 IR。内部：VM 的 lower 模块和 execute 模块分离 |
| **单一抽象级别** | ⭐⭐⭐⭐ | 外部一层（关键字），内部增加一层（块），层次清晰 |
| **可验证性** | ⭐⭐⭐⭐ | Lowering 输出是 block 格式，可独立验证（块终止指令合法性） |
| **可扩展性** | ⭐⭐⭐ | 优化 pass 在 VM 内，升级 VM 才能增加优化 |
| **概念完整性** | ⭐⭐⭐⭐ | "外部看到的是意图（关键字），内部执行的是机制（块跳转）" |

**核心优势**：
- 对外简单（只有一层关键字 IR）
- 对内可优化（内部 lowering + 块级优化）
- 降低运维复杂度（无独立 Scheduler）

**架构代价**：
- VM 职责混合（lowering + 执行），违反"一个组件只做一件事"
- 多个 VM 实例各自做 lowering，重复计算
- 前端不能检查 lowering 结果

---

## 8. 五种子方案横向对比

| 准则 | C1 关键字 | C2 二层+Scheduler | C3 单层块 | C4 Region | C5 关键字+VM内 |
|------|---------|------------------|----------|----------|------------|
| 关注点分离 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 单一抽象级别 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可验证性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 概念完整性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **综合** | **19/25** | **25/25** | **19/25** | **22/25** | **19/25** |

### 8.1 各方案的独特价值

| 方案 | 最适合场景 |
|------|-----------|
| **C1 关键字** | 快速实现、调试友好、IR 与 Redis 存储同构 |
| **C2 二层** | 多前端统一 lowering、全局优化、大型团队分工 |
| **C3 单层块** | VM 极致简单、前端单一（如只有 Python） |
| **C4 Region** | 追求与 MLIR 概念对齐、变量作用域形式化 |
| **C5 关键字+VM内** | 不愿引入独立服务、但仍需块级执行效率 |

### 8.2 关键区分维度

```
IR 复杂度 (前端负担):
  C3 > C1 = C4 = C5 > C2
  (C3 前端做 lowering, C2 前端只需结构化 IR)

VM 复杂度 (执行负担):
  C1 = C4 > C5 > C2 = C3
  (C1/C4 VM 要理解嵌套语义, C2/C3 VM 只需 br/jump)

运维复杂度:
  C2 > C1 = C3 = C4 = C5
  (C2 多一个 Scheduler 服务)

优化能力:
  C2 > C5 > C1 = C3 = C4
  (C2 有独立优化 pass, C5 在 VM 内优化, 其他不优化)
```

---

## 9. 结论

**架构最优：C2（二层 IR + Scheduler）持续领先。** 所有维度满分，代价仅是一个 Scheduler 服务。

**C1（关键字 IR）在概念完整性上同样满分。** 如果认为引入 Scheduler 的运维代价不值得——或者当前阶段不需要跨块全局优化——C1 是在架构上最自然的选择。"指令流中嵌入控制流关键字，VM 解释关键字管理嵌套作用域"这个思想简洁、直观、与 Redis 的嵌套 key 天然对齐。

**C1 和 C2 不是互斥的。** C1 可以作为 C2 的**结构化 IR 层的格式**——即：前端输出 C1 关键字 IR，Scheduler 将其 lowering 为基本块。这意味着两者可以共存演进：
1. 先实现 C1，VM 直接解释关键字，快速可用
2. 当需要全局优化时，在 VM 前插入 Scheduler（切换到 C2 或 C5）
3. VM 执行层保持不变（都能执行 br/jump 基本块）
