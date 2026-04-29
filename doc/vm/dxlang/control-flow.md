# deepxir 控制流设计方案

## 1. 三层对比：Assembly IR、deepxir、C 语言

### 1.1 函数抽象

| 维度 | Assembly IR | deepxir | C 语言 |
|------|-----------|---------|--------|
| 函数单元 | label + 指令序列 | `func name(params) -> (rets) { body }` | `ret_type name(params) { body }` |
| 调用约定 | 手动 push/pop 寄存器、手动跳转 | CALL 指令 + VM 自动管理子栈帧 | 函数调用表达式，编译器管理栈帧 |
| 参数传递 | 寄存器 / 栈偏移 | Redis KV 路径绑定（形参→实参） | 按值 / 按引用，编译器分配 |
| 返回值 | rax/eax 寄存器 或 栈 | 隐式 RETURN 将输出形参值回传父栈 | `return` 表达式 |
| 栈帧 | push rbp; sub rsp, N | `/vthread/<vtid>/<pc>/` 子键空间 | 连续栈内存 |

**deepxir 的定位**：比汇编多了一层 KV 空间抽象——函数的"栈帧"是 Redis 中的子树，参数通过路径绑定传递。比 C 少了一层编译优化——没有寄存器分配、没有 SSA、没有复杂的调用约定。

### 1.2 控制流原语

| 原语 | Assembly IR | deepxir (当前) | deepxir (目标) | C 语言 |
|------|-----------|--------------|--------------|--------|
| 顺序执行 | PC++ | `[i,0] → [i+1,0]` | 同左 | `;` 分隔 |
| 无条件跳转 | `jmp label` | — (缺失) | `jump <block>` | `goto label` |
| 条件分支 | `cmp; je/jne label` | `if cond → true/false 子树` (部分) | `if (cond) block1 else block2` | `if/else` |
| 循环 | `jmp` 回跳 | — (缺失) | `for` / `while` 展开为基本块 | `for` / `while` |
| 函数调用 | `call func` | `CALL func → eager inline 翻译` | 同左 | `func(args)` |
| 函数返回 | `ret` | 隐式 RETURN + 子栈清除 | 同左 | `return expr` |
| 多路分支 | `jmp [table]` | — | `switch` → 跳转表 | `switch/case` |

### 1.3 状态管理

| 维度 | Assembly IR | deepxir | C 语言 |
|------|-----------|---------|--------|
| 变量存储 | 寄存器 / 内存地址 | Redis key (`/vthread/<vtid>/<slot>`) | 栈 / 堆 内存 |
| 作用域 | 全局（label 可见） | vthread 子树内全局 | 词法作用域 `{}` |
| 活跃分析 | 程序员 / 编译器负责 | VM 不追踪（调用方清理子栈） | 编译器 RA + 析构 |
| 并发安全 | 原子指令 + 内存屏障 | Redis WATCH/MULTI/EXEC (picker) | mutex / atomic |

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

### 3.7 与 C 语言的关键差异总结

| 特性 | C 语言 | deepxir |
|------|--------|---------|
| 变量生命周期 | 栈帧自动管理（`{}` 作用域） | Redis key 显式管理（newtensor/deltensor） |
| 控制流嵌套 | 任意深度词法嵌套 | 翻译为平铺基本块 + jump |
| 短路求值 | `&&` `||` 编译为条件跳转 | 当前不支持（需展开为 if 嵌套） |
| 异常处理 | `try/catch` + 栈展开 | 不支持（错误标记为 vthread error） |
| 函数指针 | 支持（虚表/函数指针） | 不支持（函数名静态绑定） |
| 闭包 | 捕获环境变量 | 不支持 |

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

1. `switch/case` 多路分支
2. 短路求值 (`&&` / `||` 展开为 br 链)
3. 尾调用优化 (tail call → jump)
