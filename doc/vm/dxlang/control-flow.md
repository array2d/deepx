# deepxir 控制流设计方案

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
  gpu dialect  (GPU 抽象)            op-metal / op-cuda ← 后端调度 (隐性)
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

## 3. 设计方案：deepxir 控制流

### 3.1 核心思路

将控制流下沉为**基本块（basic block）+ 跳转（jump）**的汇编级模型，同时在 deepxir 语法层保留结构化控制流（if/for/while），翻译时展开为基本块。

```
深层原理:                          deepxir 语法层:
                               
  block0:                        if (cond) {
    inst1                          block1
    br cond, block1, block2      } else {
                                   block2
  block1:                        }
    inst2                         block3
    jump block3
                                 
  block2:
    inst3
    jump block3
                                 
  block3:  (merge point)
    inst4
```

### 3.2 基本块模型

每个基本块：
- **入口标签**：唯一的 block id（如 `@0`, `@1`, `@2`）
- **指令序列**：0 条或多条顺序指令
- **终止指令**：恰好 1 条（`br` 条件跳转 / `jump` 无条件跳转 / `return` / `call`）

#### 执行层存储格式（Redis）

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

### 3.3 结构化控制流 → 基本块翻译

#### 3.3.1 if/else → 基本块

```
源代码:
    newtensor("f32", "[4]") -> /data/a
    if (cond) {
        add(/data/a, 1) -> /data/b
    } else {
        sub(/data/a, 1) -> /data/b
    }
    deltensor(/data/a)
```

```
翻译后基本块:
    @0:                         # entry
        newtensor("f32", "[4]") -> /data/a
        br cond, @1, @2
    
    @1:                         # true branch
        add(/data/a, 1) -> /data/b
        jump @3
    
    @2:                         # false branch
        sub(/data/a, 1) -> /data/b
        jump @3
    
    @3:                         # merge
        deltensor(/data/a)
        return
```

#### 3.3.2 for 循环 → 基本块

```
源代码:
    var i <- 0
    for (i < 10) {
        add(acc, i) -> acc
        i <- i + 1
    }
```

```
翻译后基本块:
    @0:                         # init
        var i <- 0
        jump @1
    
    @1:                         # header (条件判断)
        br (i < 10), @2, @4
    
    @2:                         # body
        add(acc, i) -> acc
        i <- i + 1
        jump @1                 # 回边
    
    @4:                         # exit (循环后)
        ...
```

#### 3.3.3 while 循环 → 基本块

while 与 for 结构相同：`header → body → back-edge → header`。

```
源代码:
    while (cond) {
        body
    }
```

```
    @0:
        jump @1
    
    @1:                         # header
        br cond, @2, @3
    
    @2:                         # body
        body
        jump @1
    
    @3:                         # exit
```

### 3.4 新 opcode 定义

| Opcode | 含义 | Reads | Writes | 语义 |
|--------|------|-------|--------|------|
| `br` | 条件分支 | `[cond, true_block, false_block]` | — | if cond → PC=true_block else PC=false_block |
| `jump` | 无条件跳转 | `[target_block]` | — | PC = target_block |
| `call` | 函数调用 | `[func_name, args...]` | `[rets...]` | 创建子栈帧, PC 入子栈 |
| `return` | 函数返回 | `[ret_val]` | — | 清除当前栈帧, PC 回父栈 |
| `switch` | 多路分支 (可选) | `[value, case0_block, case1_block, ...]` | — | 跳转到匹配的 case block |

### 3.5 VM 执行循环 (新)

```
Execute(vtid):
    frame = top_frame(vtid)
    block = frame.current_block
    
    while block != nil:
        // 1. 执行 block 内所有非终止指令
        for inst in block.instructions:
            exec(inst)  // newtensor, add, deltensor, ...
        
        // 2. 执行终止指令，确定下一个 block
        switch block.terminator.opcode:
        case "br":
            cond = eval(block.terminator.reads[0])
            next_block = cond ? block.terminator.reads[1] : block.terminator.reads[2]
        case "jump":
            next_block = block.terminator.reads[0]
        case "return":
            pop_frame(vtid)
            frame = top_frame(vtid)
            if frame == nil:  mark_vthread_done(vtid); return
            next_block = frame.next_after_call
        case "call":
            push_frame(vtid, func=reads[0])
            frame = top_frame(vtid)
            next_block = frame.entry_block
        
        block = next_block
```

### 3.6 CALL/RETURN 栈帧模型 (改进)

```
当前模型:                        改进模型:
                                
/vthread/vt1/                    /vthread/vt1/
├── [0,0] = "CALL"              ├── frame:0           # 根栈帧
├── [0,-1] = "myfunc"           │   ├── @0 = entry
├── [0,0]/                       │   ├── @1 ...
│   ├── [0,0] = "add"           │   └── call myfunc → frame:1
│   └── ...                     │
├── [0,0]/[1,0]/                 ├── frame:1          # 子栈帧 (myfunc)
│   └── ...                     │   ├── @0 = entry
                                │   ├── @1 ...
嵌套路径 = 调用深度              │   └── return → pop
                                │
                                平铺路径 + frame 层次
```

**改进点**：
- block id (`@0`, `@1`) 是平面索引，不与调用层级耦合
- frame 管理独立于 block 寻址
- 同一函数的不同调用实例共享 block 结构，frame 区分上下文

### 3.7 四层关键差异总结

| 特性 | Assembly IR | deepxir | MLIR | C 语言 |
|------|-----------|---------|------|--------|
| 变量生命周期 | 手动管理 | Redis key 显式（newtensor/deltensor） | SSA + memref alloc/dealloc | 栈帧自动（`{}` 作用域） |
| 控制流嵌套 | 无限制 flat label | 翻译为平铺基本块 + jump | 结构化 ↔ CFG dialect 可互转 | 任意深度词法嵌套 |
| 短路求值 | `cmp; je` 链 | 当前不支持（需展开为 if 嵌套） | `scf.if` 内嵌 `arith.cmp` | `&&` `||` 编译为条件跳转 |
| 异常处理 | 信号/中断 | 不支持（错误标记为 vthread error） | 不内置（可自定义 dialect） | `try/catch` + 栈展开 |
| 函数指针 | `call [addr]` | 不支持（函数名静态绑定） | `func.call_indirect` | 支持（虚表/函数指针） |
| 闭包 | 无 | 不支持 | `llvm.mlir.undef` / dialect 自定义 | 捕获环境变量 |
| 可扩展性 | 新指令集扩展 | 新 opcode 注册 + 后端路由 | **dialect 注册** (核心创新) | 宏 / 内联汇编 |
| 异构计算 | 无 | op-plat 多后端路由 | gpu dialect + 目标代码生成 | CUDA/OpenCL 库调用 |
| 内存模型 | 无/arch 定义 | heap-plat 堆管理 | memref dialect + 视图/别名分析 | 指针 + malloc/free |

**MLIR 最独特的优势** — dialect 机制：在同一个 IR 框架内定义新的操作、类型、属性，无需修改核心框架。deepxir 如果要支持类似扩展性，需要在 Redis schema 层面预留 **算子分类前缀**（如 `op:linalg:`, `op:gpu:`, `op:scf:`），并允许不同 "方言" 的算子在 VM 内共存和互操作。

---

## 4. 实现路线图

### Phase 1: 基本块 + 跳转 (当前 → 结构化 CFG)

1. 引入 `jump` 和 `br` opcode
2. 修改 `Execute` 循环支持 block 跳转
3. 前端编译 if/else/for → 基本块图
4. 保持现有 `call`/`return` 机制不变

### Phase 2: 结构化语法支持

1. 前端支持 `if (cond) { ... } else { ... }` 语法
2. 前端支持 `for (init; cond; step) { ... }` 语法
3. 前端支持 `while (cond) { ... }` 语法
4. `parseDxlang` 识别控制流关键字，生成对应 block 结构

### Phase 3: 优化

1. 基本块合并（连续无条件跳转消除）
2. 死代码消除（不可达 block 清理）
3. 循环不变量外提

### Phase 4: 高级控制流（可选）

1. `switch/case` 多路分支（对标 MLIR `cf.switch`）
2. 短路求值 (`&&` / `||` 展开为 br 链)
3. 尾调用优化 (tail call → jump)

### Phase 5: MLIR 启发的能力（远期）

1. **算子分类命名空间** — Redis key 引入 dialect 前缀: `/op/linalg:<opcode>`, `/op/scf:<opcode>`, 不同方言算子可共存互操作
2. **Pass 验证** — 每个翻译阶段结束时执行 schema 验证（如 block 终止指令必须存在且唯一、SSA-like 定义-使用完整性检查）
3. **渐进式 lowering** — 支持多层深度 lowering pipeline: `结构化控制流 → CFG → 单指令 dispatch`, 每层独立可测试
4. **Dialect 注册机制** — 允许后端热注册新算子"方言"（如 `op:vision:conv2d`），route 层自动发现并调度
