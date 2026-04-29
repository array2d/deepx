# `->`/`<-` 与 SSA 编译器分析能力对比

> **核心问题**：deepxir 的 `->`/`<-` 读写分离模型，能否在不引入完整 SSA 的前提下，支持经典编译器的全部分析和优化 pass？
> **结论**：是。通过**变量版本号编码控制流合流**，以零语法增量获得等同 SSA 的完整分析能力。

---

## 1. 问题定义

```
给定一个 ->/<- 形式的 deepxir 函数:

  fn example(flag: bool, a: f32, b: f32) -> y: f32 {
    a + b -> t
    if flag {
      t * 2.0 -> x
    } else {
      t + 3.0 -> x
    }
    x + 1.0 -> y          # 写入输出参数 y 即隐式返回
  }

目标: 在不转换为完整 SSA (不引入 block argument、φ 节点、CFG 基本块) 的前提下:
  Q1: 死代码消除 — 跨分支的未使用值能否识别并删除？
  Q2: 公共子表达式消除 — 相同表达式能否跨分支识别？
  Q3: 常量传播 — 条件常量能否跨分支传播？
  Q4: 全局值编号 — 跨分支的值能否分配唯一编号？
  Q5: 循环不变量外提 — 循环内不变计算能否移到循环外？
  Q6: 寄存器分配 — 能否构建精确的干涉图？
```

## 2. SSA 与 `->`/`<-` 的本质差异

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  SSA  = 值身份模型 (Value Identity)                        │
│         每个变量有唯一静态定义点                             │
│         追踪"哪个值"被使用                                  │
│         use-def 链: 从使用点 O(1) 回溯到唯一定义点          │
│         控制流合流: block argument / φ 节点                │
│         典型系统: LLVM IR, MLIR, GCC GIMPLE                │
│                                                            │
│  ->/<- = 存储效应模型 (Storage Effect)                     │
│          变量是可复用的存储槽位                             │
│          追踪"哪个存储位置"被读写                           │
│          数据流: reads[] / writes[] 数组显式标注           │
│          控制流合流: slot 复用 (同名字覆盖)                │
│          典型系统: deepxir                                 │
│                                                            │
│  核心分歧: SSA 关心"值的身份"，->/<- 关心"存储的效应"。     │
│  两者服务不同层次，不是替代关系，是互补关系。               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

| 维度 | SSA | `->`/`<-` |
|------|-----|----------|
| **变量赋值次数** | 严格 1 次 | 无限次（可变） |
| **值标识** | 虚拟寄存器编号 (`%0`, `%1`) | KV 路径 (`./x`, `/data/W`) |
| **数据流表达** | 操作数引用 = 隐式 use-def 链 | 显式 reads[] / writes[] 数组 |
| **控制流合流** | block argument / φ 节点 | slot 复用 (同名覆盖) |
| **副作用建模** | 需隔离到 dialect (memref) | 天然 (tensor.new/del 即指令) |
| **定义-使用关系** | 支配树保证: 定义支配所有使用 | 需额外分析 (到达定义) |

### 线性段的等价性

在无控制流分支的线性代码中，两者严格等价。每个 SSA 虚拟寄存器 `%i` 与首次出现在写位置的变量名构成双射映射。

### 控制流合流 — 两种模型的分岔点

```
场景: if/else 分支后使用结果

SSA (MLIR 风格):                   ->/<- (slot 复用):
  br flag ? @then : @else            if flag {
                                     a + 1.0 -> x
@then:                             } else {
  %t = add %a, 1.0                   a - 1.0 -> x
  br @merge(%t)                    }
                                   x * 2.0 -> y
@else:
  %e = sub %a, 1.0
  br @merge(%e)

@merge(%y: f32):        ← block arg
  %r = mul %y, 2.0
  ret %r

机制: block argument 参数化        机制: slot x 被两个分支复用
优势: SSA 贯穿始终                 优势: 零额外语法，VM 直接读 slot
代价: 需 CFG + block 分割          代价: 失去值身份，编译器分析困难
```

**核心矛盾**: SSA 为编译器分析而生，但要求 CFG + block 结构。
`->`/`<-` 为执行透明而生，但丢失值身份信息。
**版本号方案同时在两种模型上取长补短。**

## 3. 方案：变量版本号编码合流

### 3.1 版本号格式

```
版本号以 @ 与变量名分隔（区别于 KV 路径的 / 分隔符）。
@ 之后用 / 分层，每级使用语义单词标记控制流来源，
合流点以 merge 显式标注。

格式:
  slot "@" base ("/" branch)*

  base   = INT                          # 根版本号 (0, 1, 2...)
  branch = "then" | "else"             # if/else 分支来源
         | "loop" | "body" | "step"    # for/while 循环来源
         | "merge"                     # 合流点 (隐式 φ)
```

### 3.2 设计原则

```
规则:

1. 每个 slot 维护独立的 base 版本计数器，从 0 开始
2. 线形写入 (-> 右侧): 产生新 base 版本 slot@N
   读取 (-> 左侧): 使用当前可见版本
3. 进入控制流分支: 在当前版本上追加分支标签 (/then /else /loop 等)
   嵌套: slot@0/then/then  (outer then → inner then)
4. 控制流合流: 自动产生合并版本 slot@N/merge = φ(分支来源...)
   合流后 base 计数器 +1 (合流结果视为新 base)

解析规则:
  最后一个 @ 之前 = 变量 KV 路径
  @ 之后 = 版本路径 (/ 分层)

示例:
  /models/W@0/then         ← 堆变量 W 的 base 0 → then 分支
  /vthread/vt1/x@0/then    ← 栈变量 x 的 base 0 → then 分支
  ./tmp@0/then/else        ← 局部变量 tmp 的 base 0 → then → else
```

**VM 行为**: VM 忽略 `@` 及之后全部内容。只看 slot 名，slot 复用语义不变。

### 3.3 语法示例

```
线性代码:
  constant(3.0) -> a@0          # a 首次写入 → base 0
  a@0 + 2.0 -> b@0              # 读 a@0, 写 b@0
  b@0 * 4.0 -> c@0              # 读 b@0, 写 c@0

带分支:
  a@0 + 1.0 -> x@0              # 合流前写入
  if flag {
    x@0 * 2.0 -> x@0/then       # "版本 0 的 then 结果"
  } else {
    x@0 * 3.0 -> x@0/else       # "版本 0 的 else 结果"
  }
  # 合流: x@0/merge = φ(x@0/then, x@0/else)
  x@0/merge + 4.0 -> y@0        # 读合流结果 → 隐式返回

嵌套分支:
  a@0 + 1.0 -> x@0
  if outer {
    if inner {
      x@0 * 2.0 -> x@0/then/then
    } else {
      x@0 * 3.0 -> x@0/then/else
    }
    # inner merge: x@0/then/merge = φ(x@0/then/then, x@0/then/else)
    x@0/then/merge + 1.0 -> x@0/then/body
  } else {
    x@0 * 4.0 -> x@0/else
  }
  # outer merge: x@0/merge = φ(x@0/then/body, x@0/else)
  x@0/merge / 2.0 -> y@0

循环:
  0.0 -> acc@0
  for i in 0..100 {
    # 循环头合流: acc@0/loop = φ(acc@0, acc@0/loop/body)
    data[i] + acc@0/loop -> acc@0/loop/body
  }
  # 循环退出: acc@0/merge = φ(acc@0, acc@0/loop/body)
  acc@0/merge -> total@0
```

### 3.4 编译器合流识别

```
merge 标签使合流识别退化为前缀匹配:

判定规则:
  1. /merge 后缀 → 直接识别为合流版本
  2. 合流来源 = 去掉末尾 /merge 后, 以前缀匹配的所有分支版本
     例: x@0/then/merge → 前缀 x@0/then/ → 来源 = x@0/then/then, x@0/then/else
     例: x@0/merge → 前缀 x@0/ → 来源 = x@0/then, x@0/else, x@0/then/body
  3. 若某分支无写入 → 来源为 base 版本

算法:

  func resolve_merge(version_str):
    slot, ver_path = parse(version_str)   // "x@0/then/merge" → slot="x", path=[0,then,merge]
    if not ver_path.ends_with("merge"):
      return  // 非合流版本

    // 提取前缀: "x@0/then/merge" → "x@0/then/"
    prefix = version_str.strip_suffix("/merge") + "/"

    // 收集所有匹配前缀的版本 → φ 来源集合
    sources = find_all_versions_with_prefix(prefix)

    // 补全: 某分支无写入 → 来源 = base 版本
    for branch in get_branches_from_cfg():
      if no version matches prefix + branch:
        sources.append(base_version(slot, prefix))

    return create_phi(version_str, sources)
```

## 4. 编译器分析能力验证

### 4.1 死代码消除 (DCE)

```
a@0 + 1.0 -> x@0
if flag {
  x@0 * 2.0 -> x@0/then
  b@0 * 3.0 -> dead@0/then     ← 仅在 then 分支被定义
} else {
  x@0 * 4.0 -> x@0/else
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 3.0 -> y@0

分析: dead@0/then 从未被读 → 死代码 → 删除 ✅
```

### 4.2 公共子表达式消除 (CSE)

```
if flag {
  a@0 + b@0 -> x@0/then        ← add(a@0, b@0)
} else {
  a@0 + b@0 -> x@0/else        ← 相同操作数和操作码!
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge * 2.0 -> y@0

分析: x@0/then 和 x@0/else 的定义完全一致 → 同一值
     φ 退化为单一来源 → 消除冗余定义 ✅
```

### 4.3 全局值编号 (GVN)

```
a@0 + b@0 -> t@0
c@0 + d@0 -> u@0
if flag {
  a@0 + b@0 -> x@0/then        ← VN = 42 (同 t@0)
} else {
  c@0 + d@0 -> x@0/else        ← VN = 17 (同 u@0)
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 1.0 -> y@0

分析: t@0=42, u@0=17, x@0/then=42, x@0/else=17
     x@0/merge = φ(42, 17) → VN = hash(φ, 42, 17) ✅
```

### 4.4 稀疏条件常量传播 (SCCP)

```
constant(true) -> flag@0
if flag@0 {
  constant(5.0) -> x@0/then
} else {
  constant(3.0) -> x@0/else    ← 不可达! flag=true
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 1.0 -> y@0

分析: flag@0=true → then 可达到, else 不可达
     x@0/then=5.0, x@0/else=⊥
     x@0/merge = φ(5.0, ⊥) = 5.0 → y@0 = 6.0 ✅
```

### 4.5 循环不变量外提 (LICM)

```
a@0 * b@0 -> inv@0          ← 定义在循环外
0.0 -> acc@0
for i in 0..n {
  # 循环头合流: acc@0/loop = φ(acc@0, acc@0/loop/body)
  inv@0 + acc@0/loop -> acc@0/loop/body
}

分析: inv@0 定义在循环外, 循环内无新版本 → 循环不变量 ✅
```

### 4.6 寄存器分配 / 干涉图

```
a@0 + b@0 -> t1@0
t1@0 * 2 -> t2@0
if flag {
  t1@0 + 5 -> t3@0/then
} else {
  t2@0 + 3 -> t3@0/else
}
# 合流: t3@0/merge = φ(t3@0/then, t3@0/else)
t3@0/merge * t1@0 -> z@0

分析: 每个版本有唯一活性区间
     t3@0/then 与 t3@0/else 不干涉 (不同分支)
     t1@0 与 t3@0/merge 干涉 → 图着色精确 ✅
```

## 5. 与纯 SSA 的架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  纯 SSA (MLIR cf dialect):                                      │
│                                                                 │
│  @0:                                                            │
│    %0 = add %a, %b                                              │
│    br %flag ? @1 : @2              ← 基本块分割                 │
│  @1:                                                            │
│    %1 = add %a, 2.0                                              │
│    br @3(%1)                       ← block argument 传递        │
│  @2:                                                            │
│    %2 = sub %a, 2.0                                              │
│    br @3(%2)                       ← block argument 传递        │
│  @3(%x: f32):                      ← %x = φ(%1, %2)            │
│    %3 = mul %x, 3.0                                              │
│    ret %3                                                        │
│                                                                 │
│  版本号编码合流 (deepxir):                                        │
│                                                                 │
│  fn example(flag: bool, a: f32) -> y: f32 {                     │
│    a@0 + b@0 -> x@0                                              │
│    if flag {                                                     │
│      a@0 + 2.0 -> x@0/then          ← base 0 → then 分支        │
│    } else {                                                      │
│      a@0 - 2.0 -> x@0/else          ← base 0 → else 分支        │
│    }                                                            │
│    x@0/merge * 3.0 -> y@0            ← x@0/merge = φ(x@0/then, x@0/else) │
│  }                                ↑ 写入 y@0 = 隐式返回         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────┬──────────────────────┬──────────────────────┐
│ 维度                    │ 纯 SSA               │ 版本号编码合流        │
├────────────────────────┼──────────────────────┼──────────────────────┤
│ IR 结构变化             │ 函数→基本块→指令     │ 函数→指令(一层)      │
│ 合流机制                │ block argument       │ 版本号隐式编码        │
│ 前端生成复杂度          │ 高 (CFG 构造 + φ)    │ 低 (版本计数器)      │
│ VM 执行                 │ 需 block-loop        │ slot 复用,零变化      │
│ 编译分析能力            │ 原生 SSA             │ 内部 SSA (自动构建)  │
│ 向后兼容                │ ❌ 不兼容现有 IR      │ ✅ 不加版本号照常运行 │
│ 人可读性                │ 编号 %0 %1 无意义    │ 名字有意义 + 版本可溯源│
│ 与执行协议对齐           │ 需 lower 到 slot     │ 直接 = slot 格式     │
└────────────────────────┴──────────────────────┴──────────────────────┘
```

## 6. 实现路线

```
Phase 1: 前端版本号生成
  ┌─────────────────────────────────────────────┐
  │ 每个 slot 维护: slot → current_base          │
  │ 线形写入: 输出 "slot@N"                      │
  │ 进入分支: 追加分支标签 "slot@N/then" 等       │
  │ 嵌套分支: 逐级追加 "/then" "/else"           │
  │ 合流点: 自动产生 "slot@N/merge"              │
  │                                              │
  │ 新增代码量: ~100-200 行                      │
  │ VM: 0 行 (忽略 @ 及之后全部内容)             │
  └─────────────────────────────────────────────┘

Phase 2: 编译器合流识别
  ┌─────────────────────────────────────────────┐
  │ 解析 slot@path 格式 (@ 分隔 slot 与版本)     │
  │ /merge 后缀直接识别为合流版本                 │
  │ 前缀匹配查找 φ 来源集合 (无需回溯 CFG)       │
  │                                              │
  │ 新增代码量: ~150-250 行                      │
  └─────────────────────────────────────────────┘

Phase 3: 优化 Pass 全线启用
  ┌─────────────────────────────────────────────┐
  │ 内部 SSA 已就绪                               │
  │ 复用标准优化 pass (DCE, CSE, GVN, SCCP...)   │
  │ 优化结果映射回 slot → 输出优化后 IR           │
  └─────────────────────────────────────────────┘
```

## 7. 结论

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Q: ->/<- 能否替代 SSA 做编译分析的全部需求？                 │
│                                                              │
│  A: 能。通过变量版本号编码控制流合流。                        │
│                                                              │
│  纯 slot 复用:                                               │
│    ❌ 缺少值身份 → 全局分析不可行                            │
│    ✅ 执行层高效 (VM 直接读写 slot)                          │
│                                                              │
│  版本号编码合流:                                              │
│    ✅ 线性段: slot@ver 直接提供 use-def 链, O(1)             │
│    ✅ 控制流合流: /merge 触发自动 φ 构建                    │
│    ✅ 全部 pass: DCE, CSE, GVN, SCCP, LICM, 寄存器分配     │
│    ✅ 零新语法: 版本号即后缀, 无关键字, 无 block arg         │
│    ✅ VM 零变化: slot 复用保持不变, @ 之后全部忽略           │
│    ✅ 向后兼容: 不加版本号照常运行 (无优化)                  │
│                                                              │
│  这不是"用 ->/<- 替代 SSA"。                                 │
│  这是"把 SSA 的编译分析能力折叠进 ->/<- 的版本号系统中"。     │
│                                                              │
│  版本号 = SSA 的值身份                                      │
│  slot = 存储实体                                            │
│  /merge = φ 的显式编码                                      │
│  结构化控制流 = CFG 的确定性描述                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

> **关联文档**:
> - [spec-control-flow-v1.md](spec-control-flow-v1.md) — `->`/`<-` 与 SSA 语义层面对比 + C1~C5 方案架构
> - [control-flow.md](control-flow.md) — 控制流 IR 设计（基本块模型）
> - [frontend-control-flow.md](frontend-control-flow.md) — 前端代码生成方案
