# deepxir 控制流设计

> **方案分析**：[spec-control-flow-v1.md](spec-control-flow-v1.md) — 5 种 C 子方案架构对比。
> 本文提供 MLIR 对比 + 基本块模型 + C1 关键字 IR 具体设计。

## 1. 四层对比：Assembly IR、deepxir、MLIR、C 语言

> **MLIR** (Multi-Level Intermediate Representation) 与 deepxir 的设计目标高度重合：
> 都面向深度学习计算场景，都需要在计算图和指令级之间提供可组合的中间表示。
> deepxir 可以视为 MLIR 在 **KV 存储原生、分布式调度、多后端异构** 方向上的一个垂直领域方言实现。

### 1.1 函数抽象

| 维度 | Assembly IR | deepxir | MLIR | C 语言 |
|------|-----------|---------|------|--------|
| 函数单元 | label + 指令序列 | `def name(params) -> (rets) { body }` | `func.func @name(%arg: T) -> T { body }` | `ret_type name(params) { body }` |
| 调用约定 | 手动 push/pop 寄存器、手动跳转 | CALL 指令 + VM 自动管理子栈帧 | `func.call @callee(%args)` + 显式 SSA 结果 | 函数调用表达式，编译器管理栈帧 |
| 参数传递 | 寄存器 / 栈偏移 | Redis KV 路径绑定（形参→实参） | Block arguments (SSA 值), 支持 memref 传递 | 按值 / 按引用，编译器分配 |
| 返回值 | rax/eax 寄存器 或 栈 | 隐式 RETURN 将输出形参值回传父栈 | `func.return %val : T` 显式返回 SSA 值 | `return` 表达式 |
| 栈帧 | push rbp; sub rsp, N | `/vthread/<vtid>/<pc>/` 子键空间 | Region + block hierarchy, alloc 可下沉 | 连续栈内存 |
| 类型系统 | 无 (原始字节) | 签名标注型别，运行时类型感知求值 | 完整类型体系 (tensor/memref/vector/...) | 编译期静态类型 |

**deepxir 的定位**：在汇编之上抽象了 KV 空间——函数帧是 Redis 子树，参数通过路径绑定传递。与 MLIR 共享"深度学习 IR"的目标，但 MLIR 侧重编译优化（SSA、dialect 混用、pass pipeline），deepxir 侧重运行时调度（Redis 原生、多后端路由、vthread 并发）。比 C 少一层编译复杂度，比 MLIR 少一层类型和 SSA 的抽象约束。

### 1.2 控制流原语

| 原语 | Assembly IR | deepxir (当前) | deepxir (目标) | MLIR | C 语言 |
|------|-----------|--------------|--------------|------|--------|
| 顺序执行 | PC++ | `[i,0] → [i+1,0]` | 同左 | block 内顺序 op | `;` 分隔 |
| 无条件跳转 | `jmp label` | — (缺失) | `jump <block>` | `cf.br ^block` | `goto label` |
| 条件分支 | `cmp; je/jne label` | `if cond → true/false 子树` (部分) | `if (cond) block1 else block2` | `scf.if` / `cf.cond_br ^t, ^f` | `if/else` |
| 循环 | `jmp` 回跳 | — (缺失) | `for` / `while` 展开为基本块 | `scf.for` / `scf.while` / `affine.for` | `for` / `while` |
| 函数调用 | `call func` | `CALL func → eager inline 翻译` | 同左 | `func.call @callee(%args)` | `func(args)` |
| 函数返回 | `ret` | 隐式 RETURN + 子栈清除 | 同左 | `func.return %val` | `return expr` |
| 多路分支 | `jmp [table]` | — | `switch` → 跳转表 | `cf.switch` / `scf.index_switch` | `switch/case` |
| 结构化抽象 | — | — | if/for 语法糖翻译为基本块 | `scf.` dialect (高层) ↔ `cf.` dialect (底层) | 原生结构化 |

### 1.3 状态管理

| 维度 | Assembly IR | deepxir | MLIR | C 语言 |
|------|-----------|---------|------|--------|
| 变量存储 | 寄存器 / 内存地址 | Redis key (`/vthread/<vtid>/<slot>`) | SSA Value (虚拟寄存器编号) → 可下沉到 memref | 栈 / 堆 内存 |
| 作用域 | 全局（label 可见） | vthread 子树内全局 | Block/Region 隔离, SSA 支配规则 | 词法作用域 `{}` |
| 活跃分析 | 程序员 / 编译器负责 | VM 不追踪（调用方清理子栈） | SSA use-def chain 自动推导, 编译器负责 | 编译器 RA + 析构 |
| 并发安全 | 原子指令 + 内存屏障 | Redis WATCH/MULTI/EXEC (picker) | 显式 async/await dialect, gpu.async 等 | mutex / atomic |
| 数据流 | 隐式 (side effect) | 显式 (reads/writes 路径数组) | 显式 SSA (操作数 ↔ 结果 编号链接) | 隐式 + 指针别名 |

**MLIR 的 SSA 模型**是其核心设计：每个值有唯一编号（`%0`, `%1`），use-def 链天然形成数据流图。deepxir 的数据流则通过 Redis key 路径（`reads`/`writes` 数组）显式编码，适合分布式场景，但缺少编译期 use-def 验证的严格性。

### 1.4 MLIR 核心创新对 deepxir 的启发

MLIR 的三个核心设计对 deepxir 有直接参考价值：

#### Dialect 方言体系 vs deepxir 算子分类

```
MLIR:                              deepxir (类比):
                               
  func dialect  (函数定义)          def ... -> () { }  ← 函数层
  scf dialect  (结构化控制流)        if / for / while  ← 控制流层 (目标)
  cf dialect   (底层控制流)          br / jump / switch ← CFG 层 (目标)
  linalg dialect (线性代数)          matmul, add, conv  ← 计算层
  gpu dialect  (GPU 抽象)            exop-metal / op-cuda ← 后端调度 (隐性)
  memref dialect (内存抽象)           newtensor/deltensor ← 生命周期层
  arith dialect (算术)               + - * / % + 原生算子 ← 求值层
```

MLIR 的 **dialect 混用** 允许在同一函数中混合不同抽象层的操作（如在 `scf.for` 循环内调用 `linalg.matmul`），deepxir 当前已经隐式实现了类似能力——生命周期 (`newtensor`)、计算 (`add`)、控制 (`if`) 可以在同一函数体中混用。**但 deepxir 缺少 dialect 的显式声明和校验机制**。

#### Region / Block 层级模型

```
MLIR:                              deepxir (目标):
                               
  func @main {                     def main(...) -> (...) {
    ^bb0(%arg0: f32):                @0:                      ← entry block
      %0 = arith.addf %arg0, %cst    newtensor(...) -> /data
      cf.br ^bb1(%0)                 br ... → @1/@2
                                    
    ^bb1(%1: f32):                  @1:                      ← loop header
      scf.for %i = %lb to %ub {      ... (block body)
        %2 = linalg.matmul ...       
        ...
      }
  }                                }
```

MLIR 的 **Region** 是其关键抽象——`scf.for` 的循环体是一个嵌套的 Region，与父函数的 value 作用域隔离。deepxir 当前用 **PC 路径嵌套** (`/vthread/<vtid>/<pc>/`) 实现了类似隔离，但控制流跳转（jump）跨越嵌套时需要更明确的 frame/region 边界语义。

#### Pass Pipeline 与 deepxir 的翻译阶段

```
MLIR 编译流程:                    deepxir 当前 + 目标流程:
                               
  前端 (C++/PyTorch)      →       pysdk 写入 /src/func/ (dxlang)
    │                                │
  dialect lowering          →       ParseDxlang → ir.Instruction (语法→结构)
    │                                │
  canonicalization           →       eager inline translate (形参绑定, 坐标化)
    │                                │
  loop/affine optimization   →       基本块合并 / 死代码消除 (Phase 3 目标)
    │                                │
  gpu mapping                →       route.Select(backend) → op-plat
    │                                │
  LLVM lowering              →       heap-plat 生命周期指令
    │                                │
  机器码                     →       GPU kernel 执行
```

deepxir 的当前流程已经是一条隐式的 "pass pipeline"，但缺少 MLIR 的 **可组合性** 和 **可验证性**：每个 pass 的输入输出格式没有形式化定义，变换的正确性依赖人工保证而非结构化约束。

---

## 2. deepxir 当前控制流状态

### 2.1 已实现

```
控制流 opcode: call, return, if

call  → translate.HandleCall()
  - eager inline: 一次性将编译层 dxlang 翻译为执行层 [i,j] 坐标
  - 形参/实参绑定: replaceParams(parsed.Reads/Writes, bindings)
  - 子栈根: /vthread/<vtid>/<pc>/
  - 隐式 return 指令: 追加在子栈末尾

return → translate.HandleReturn()
  - 返回值回传: 读取 retRef, 写入父栈 retSlot
  - 子栈清除: KEYS + DEL
  - PC 恢复: NextPC(parentPC)

if    → dispatch.If()
  - 条件求值: isTruthy(condVal) → true / false
  - 分支 PC: pc+"/true/0" 或 pc+"/false/0"
  - 无合并点: 分支末尾直接进入 nextPC 或 done
```

### 2.2 缺失

| 缺失项 | 影响 |
|------|------|
| **无条件跳转 (`jump`)** | 无法实现循环回边、无法实现 goto-like 控制流 |
| **循环 (`for`/`while`)** | 循环只能通过前端展开为线性指令（如文档示例中的 python-sdk 写入） |
| **分支合并 (join/phi)** | if/else 无合并基本块，两个分支各自独立结束 |
| **结构化块 (block)** | 控制流没有明确的"基本块"边界，靠子树路径区分 |
| **switch/case** | 多路分支无支持 |

---

## 3. 设计方案：基本块模型（C 方案通用基础）

> C1~C5 五种方案共享核心的基本块模型，区别在于 lowering 位置和 IR 层数。
> 详见 [spec-control-flow-v1.md](spec-control-flow-v1.md)。

### 3.1 C 方案核心思想

> **控制流从"用户意图"到"机器执行"是一个语义等价的格式变换过程。**

不同 C 子方案对"几层 IR"和"lowering 在哪里"有不同选择：

| 子方案 | IR 层数 | Lowering 位置 |
|--------|--------|-------------|
| C1 关键字 | 1 层 | 无（VM 直接解释） |
| C2 二层+Scheduler | 2 层 | Scheduler 服务 |
| C3 单层基本块 | 1 层 | 前端负责 |
| C4 Region | 1 层 | VM 原生执行 region |
| C5 关键字+VM内 | 1 层(外)/2 层(内) | VM 加载时内部 lowering |

**C1 和 C2 是互补的**：C1 可作为 C2 的结构化 IR 层格式——先 C1 快速可用，需要全局优化时加 Scheduler 切换到 C2。

### 3.2 基本块模型（所有 C 方案的执行层基础）

每个基本块：
- **入口标签**：唯一的 block id（如 `@0`, `@1`, `@2`）
- **指令序列**：0 条或多条顺序指令
- **终止指令**：恰好 1 条（`br` 条件跳转 / `jump` 无条件跳转 / `return` / `call`）

### 3.3 执行层存储格式（Redis）

```
/vthread/<vtid>/<pc>/@0           → "br"        # block 0 的终止指令
/vthread/<vtid>/<pc>/@0/-1        → "cond"      # br 的条件变量
/vthread/<vtid>/<pc>/@0/-2        → "@1"        # true 目标
/vthread/<vtid>/<pc>/@0/-3        → "@2"        # false 目标

/vthread/<vtid>/<pc>/@0/0         → "newtensor" # block 0 的指令
/vthread/<vtid>/<pc>/@0/0,1       → "/data/a"
...

/vthread/<vtid>/<pc>/@1           → "jump"      # block 1 的终止指令
/vthread/<vtid>/<pc>/@1/-1        → "@3"        # 目标 block
```

#### 对比当前格式

```
当前: /vthread/<vtid>/[0,0]/[0,0]  → "newtensor"   (嵌套路径 = 子树)
                                        ^^^^^^^^
                                        PC 路径嵌套表示子栈

方案: /vthread/<vtid>/@0            → block 元数据
      /vthread/<vtid>/@0/0          → block 0 第 0 条指令
      /vthread/<vtid>/@0/0,1        → block 0 第 0 条指令第 1 个 write
```

**优点**：
- block id 是平面索引（`@0`, `@1`），不是嵌套路径（`[0,0]/[1,0]`）
- PC 不再需要 `/` 分隔符解析层级——层级由 CALL/RETURN 隐式管理
- 控制流图更清晰：block 有明确的入边和出边

### 3.4 控制流 IR：两种表达方式

#### 3.4.1 C1/C5 风格：关键字 IR（结构化）

控制流以关键字形式嵌入指令流，VM 直接解释（C1）或加载时 lowering（C5）。

```
def example(x) -> (y) {
    newtensor("f32", "[4]") -> ./a
    sum(./x) -> ./s

    if (greater(./s, 0)) {              ← 关键字
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

**Redis 存储**（嵌套 key 天然表达控制流层次）：

```
/vthread/<vtid>/[1,0]         = "if"
/vthread/<vtid>/[1,0]/cond    = "./cond"
/vthread/<vtid>/[1,0]/then/0  = "add"
/vthread/<vtid>/[1,0]/else/0  = "mul"

/vthread/<vtid>/[2,0]         = "for"
/vthread/<vtid>/[2,0]/init/0  = "var"
/vthread/<vtid>/[2,0]/cond    = "less(./i, 10)"
/vthread/<vtid>/[2,0]/step/0  = "add(./i, 1)"
/vthread/<vtid>/[2,0]/body/0  = "add(./y, ./i)"
```

#### 3.4.2 C2/C3 风格：基本块 IR（平铺）

结构化控制流 lowering 为基本块 + 终止指令。

```
@0:
    newtensor("f32", "[4]") -> ./a
    sum(./x) -> ./s
    greater(./s, 0) -> ./cond
    br ./cond, @1, @2

@1:
    add(./x, 1.0) -> ./y
    jump @3

@2:
    mul(./x, -1.0) -> ./y
    jump @3

@3:
    deltensor(./a)
    return(./y)
```

### 3.5 控制流关键字定义（C1 专用）

| 关键字 | 语义 | VM 行为 |
|--------|------|--------|
| `if (cond) { ... } [else { ... }]` | 条件分支 | 求值 cond → 进入 then 或 else 子作用域 |
| `for (init; cond; step) { ... }` | 计数循环 | 执行 init → 循环: 求值 cond → 执行 body → 执行 step |
| `while (cond) { ... }` | 条件循环 | 循环: 求值 cond → 执行 body |
| `loop { ... }` | 无限循环 | 循环 body，直到内部 break |
| `break` | 跳出最近循环 | 跳出当前循环作用域 |
| `continue` | 跳过本次迭代 | 跳到循环条件判断 |
| `switch (val) { case v1: ... default: ... }` | 多路分支 | 匹配 val 到 case |

### 3.6 终止指令定义（C2/C3 专用）

| Opcode | 含义 | Reads | 语义 |
|--------|------|-------|------|
| `br` | 条件分支 | `[cond, true_block, false_block]` | if cond → PC=true_block else PC=false_block |
| `jump` | 无条件跳转 | `[target_block]` | PC = target_block |
| `call` | 函数调用 | `[func_name, args...]` | 创建子栈帧, PC 入子栈 |
| `return` | 函数返回 | `[ret_val]` | 清除当前栈帧, PC 回父栈 |

### 3.7 VM 执行循环

**C1/C4 模式**（VM 直接解释关键字/region）：

```go
func Execute(vtid) {
    inst := decode(pc)
    switch inst.Opcode {
    case "if":
        pushScope(vtid, eval(inst.Cond) ? inst.Then : inst.Else)
    case "for":
        pushLoopScope(vtid, inst.Init, inst.Cond, inst.Step, inst.Body)
    case "while":
        pushLoopScope(vtid, nil, inst.Cond, nil, inst.Body)
    case "break":
        popLoopScope(vtid)
    case "return":
        popFrame(vtid)
    default:
        dispatch(inst)  // op-plat / heap-plat / VM 求值
    }
}
```

**C2/C3 模式**（block-loop，最低 VM 复杂度）：

```go
func Execute(vtid) {
    block := entryBlock
    for block != nil {
        for _, inst := range block.Instructions {
            dispatch(inst)
        }
        switch block.Terminator.Opcode {
        case "br":
            block = eval(terminator.Cond) ? loadBlock(terminator.True) : loadBlock(terminator.False)
        case "jump":
            block = loadBlock(terminator.Target)
        case "return":
            popFrame(); block = parent.AfterCall
        case "call":
            pushFrame(terminator.Func); block = newFrame.Entry
        }
    }
}
```

### 3.8 栈帧模型

```
/vthread/vt1/
├── frame:0                    # 根栈帧
│   ├── @0 = entry block
│   ├── @1 ...
│   └── call myfunc → frame:1
│
├── frame:1                    # 子栈帧 (myfunc)
│   ├── @0 = entry block
│   ├── @1 ...
│   └── return → pop
│
平铺路径 + frame 层次分离
block id (@0, @1) 是平面索引，不与调用层级耦合
```

---

## 4. 实现路线图

### Phase 1: C1 关键字 IR（快速可用）

1. 引入控制流关键字 opcode：`if`, `for`, `while`, `break`, `continue`, `switch`
2. VM 实现嵌套作用域栈（scope stack），直接解释关键字
3. 前端 `@compile` 生成关键字 IR
4. Redis 存储同构（嵌套 key）

### Phase 2: 基本块执行（为 C2/C5 做准备）

1. 引入 `jump`/`br` 终止指令
2. VM 实现 block-loop 执行模式（与关键字模式并存）
3. 关键字 → 基本块 lowering（在 VM 内或 Scheduler 内）
4. 前端可选择生成关键字 IR 或基本块 IR

### Phase 3: Scheduler 服务（C2）

1. lowering 逻辑从 VM 中抽出，放入独立 Scheduler
2. Scheduler 实现：Region Flattening → 块内 SSA → 优化 pass
3. 多前端（Python/Go）通过 Scheduler 统一 lowering

### Phase 4: 优化与高级控制流

1. 死代码消除、基本块合并、循环不变量外提
2. 短路求值 (`&&`/`||` 展开为 br 链)
3. 尾调用优化 (tail call → jump)
4. Dialect 命名空间预留 (`op:linalg:`, `op:scf:`)
