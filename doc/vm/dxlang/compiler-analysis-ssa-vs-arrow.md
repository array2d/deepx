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

  Q1: 死代码消除 — t 的值在 if/else 之后是否还被使用？
  Q2: 公共子表达式消除 — 两处相同的 a+b 能否识别并消除？
  Q3: 常量传播 — 条件常量能否跨分支传播？
  Q4: 全局值编号 — 能否为跨分支的值分配唯一编号？
  Q5: 循环不变量外提 — 能否识别和移动循环不变量？
  Q6: 寄存器分配 — 能否构建精确的干涉图？
```

## 2. 两种模型的本质差异

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
| **线形段等价性** | 等价 | 等价（可建立双射映射） |
| **控制流等价性** | 语义等价，结构不等价 | 语义等价，结构不等价 |

### 线形段的等价性

在无控制流分支的线性代码中，两者严格等价：

```
SSA:                              ->/<-:
  %0 = add %a, %b                   a + b -> t
  %1 = mul %0, 2.0                  t * 2.0 -> x
  %2 = add %1, 1.0                  x + 1.0 -> y       # 写入 y = 隐式返回
  ret %2

映射: 每个 SSA 虚拟寄存器 %i ←→ 首次出现在写位置的变量名
     SSA use ←→ 读位置的变量名
     双射关系: 线性段内完全一一对应
     deepxir 无 ret <value>: 写入输出参数即返回
```

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

**核心矛盾**: SSA 为编译器分析而生 (值身份、use-def、支配树)，但改变了 IR 结构。
`->`/`<-` 为执行透明而生 (slot 复用、协议直通)，但丢失了值身份信息。

## 3. 逐个 Pass 可行性分析

### 3.1 无版本号（纯 slot 复用）下的能力

| Compiler Pass | 可行性 | 需要额外什么 | 精度损失 |
|---------------|--------|-------------|---------|
| **常量折叠** | ✅ 直接 | 无 | 无 |
| **死代码消除** | ⚠️ 可行 | 活性分析 O(n²) | 无 |
| **常量传播** | ⚠️ 可行 | 到达定义分析 | 轻微 |
| **复写传播** | ⚠️ 可行 | 到达定义检查 | 无 |
| **CSE** | ⚠️ 部分可行 | 到达定义 (线性段 O(n)，控制流 O(n²)) | 有（控制流） |
| **GVN** | ❌ 不可行 | 需先 SSA 构造 | — |
| **LICM** | ⚠️ 可行 | 到达定义 + 循环识别 | 无 |
| **SCCP** | ❌ 不可行 | 需支配树 + SSA 稀疏表示 | — |
| **寄存器分配** | ⚠️ 粗略可行 | 区分变量版本 (活性分析) | 有 |
| **归纳变量** | ✅ 可行 | 模式匹配 | 无 |
| **内联** | ✅ 可行 | 无 | 无 |

**关键观察**: 几乎所有"⚠️ 可行但需额外工作"的场景，额外工作 = **到达定义分析**。
而到达定义分析的结果 = **为每个 slot 的每次写入分配唯一标识** — 这就是在做 SSA 构造。

### 3.2 失败的根源：值身份的缺失

```
为什么纯 slot 复用下的 CSE 不可靠:

  add(a, b) -> t1      ← 定义 t1
  add(a, b) -> t2      ← 也是 add(a, b)！但 t1 == t2 ?
                        ← 不确定: a, b 在两指令间可能被重定义
                        ← 编译器必须做到达定义分析才能确认

为什么纯 slot 复用下的 GVN 不可行:

  GVN 的核心: "值相等"判断 → 需要"值"这个概念
  纯 slot 复用没有"值" — 只有"slot 的当前内容"
  "slot 的当前内容"随时间变化, 无法分配稳定的值编号

结论: 纯 slot 复用缺少一类关键信息——
      "此时 slot S 中存储的值是第几次写入产生的"。
      即: 缺少**值身份**。
```

## 4. 方案：变量版本号编码合流

### 4.1 版本号格式

```
版本号以 @ 与变量名分隔（区别于 KV 路径的 / 分隔符）。
多级版本以 : 连接，每级使用语义单词标记控制流来源，
合流点以 merge 显式标注。

格式:
  slot "@" base ("/" branch)*

  base   = INT                          # 根版本号 (0, 1, 2...)
  branch = "then" | "else"             # if/else 分支来源
         | "loop" | "body" | "step"    # for/while 循环来源  
         | "merge"                     # 合流点 (隐式 φ)

设计原则:
  - @ 标记版本域起点:  /data/W@0/then  = "堆变量 W 版本 0 → then 分支"
  - @ 之后 / 为版本层级:  x@0/then  直读为 "x 在版本 0 的 then 分支"
  - merge 显式标注合流: x@0/merge 直读为 "x 在版本 0 的合流结果"
  - 嵌套自然可读:  x@0/then/else = "版本 0 → outer.then → inner.else"
  - 全单词, 无缩写: then/else/loop/body/step/merge
```

### 4.2 核心思想

**给每次写入附加版本号。控制流分支自动产生带来源标签的子版本，合流自动产生 merge 版本。
版本号 = 值身份 + 控制流溯源。merge 版本 = 隐式 φ。**

```
规则:

1. 每个 slot 维护独立的 base 版本计数器，从 0 开始
2. 线形写入 (-> 右侧): 产生新 base 版本 slot@N
   读取 (-> 左侧): 使用当前可见版本
3. 进入控制流分支: 在当前版本上追加分支标签 (:then / :else / :loop 等)
   嵌套: slot@0/then/then  (outer then → inner then)
4. 控制流合流: 自动产生合并版本 slot@N:merge = φ(分支来源...)
   合流后 base 计数器 +1 (合流结果视为新 base)
```

### 4.3 语法示例

```
线性代码:
  constant(3.0) -> a@0          # a 首次写入 → base 0
  a@0 + 2.0 -> b@0              # 读 a@0, 写 b@0
  b@0 * 4.0 -> c@0              # 读 b@0, 写 c@0

带分支:
  a@0 + 1.0 -> x@0              # 合流前写入 → base 0
  if flag {
    x@0 * 2.0 -> x@0/then       # then 分支: "版本 0 的 then 结果"
  } else {
    x@0 * 3.0 -> x@0/else       # else 分支: "版本 0 的 else 结果"
  }
  # 合流: x@0/merge = φ(x@0/then, x@0/else)
  x@0/merge + 4.0 -> y@0        # 读合流结果, 写入输出参数 y@0 = 隐式返回

嵌套分支:
  a@0 + 1.0 -> x@0
  if outer {
    if inner {
      x@0 * 2.0 -> x@0/then/then    # outer then → inner then
    } else {
      x@0 * 3.0 -> x@0/then/else    # outer then → inner else
    }
    # inner merge: x@0/then/merge = φ(x@0/then/then, x@0/then/else)
    x@0/then/merge + 1.0 -> x@0/then/body
  } else {
    x@0 * 4.0 -> x@0/else           # outer else
  }
  # outer merge: x@0/merge = φ(x@0/then/body, x@0/else)
  x@0/merge / 2.0 -> y@0

可读性对比:
  旧: x#0.t.e        →  阅读时脑内翻译 "dot-t-dot-e" → 难读
  新: x@0/then/else  →  直接读 "版本 0 的 then 分支的 else 分支"

循环:
  0.0 -> acc@0
  for i in 0..100 {
    # 循环头合流: acc@0/loop = φ(acc@0, acc@0/loop/body)
    data[i] + acc@0/loop -> acc@0/loop/body
  }
  # 循环退出: acc@0/merge = φ(acc@0, acc@0/loop/body)
  acc@0/merge -> total@0
```

**VM 行为**: VM 忽略 `@` 及之后全部内容。只看 slot 名，slot 复用语义不变。

### 4.4 编译器合流识别

```
merge 标签使合流识别退化为前缀匹配，无需回溯控制流结构:

判定规则:
  1. /merge 后缀 → 直接识别为合流版本
  2. 合流来源 = 去掉末尾 /merge 后, 以前缀匹配的所有分支版本
     例: x@0/then/merge → 前缀 x@0/then/ → 来源 = x@0/then/then, x@0/then/else
     例: x@0/merge → 前缀 x@0/ → 来源 = x@0/then, x@0/else, x@0/then/body
  3. 若某分支无写入 → 来源为进入控制流前的版本 (base 版本)

算法:

  func resolve_merge(version_str):
    slot, ver_path = parse(version_str)   // "x@0/then/merge" → slot="x", path=[0,then,merge]

    if not ver_path.ends_with("merge"):
      return  // 非合流版本

    // 提取前缀: "x@0/then/merge" → "x@0/then/"
    prefix = version_str.strip_suffix("/merge") + "

    // 收集所有匹配前缀的版本 → φ 来源集合
    sources = find_all_versions_with_prefix(prefix)

    // 补全: 某分支无写入 → 来源 = base 版本
    expected_branches = get_branches_from_cfg()
    for branch in expected_branches:
      if no version matches prefix + branch:
        sources.append(base_version(slot, prefix))

    return create_phi(version_str, sources)
```

### 4.5 版本号中的 @ 与 / 共存

```
KV 路径使用 / 分隔层级, 版本号使用 @ 分隔变量与版本:

  /models/W@0              ← 堆变量 W 的 base 版本 0
  /vthread/vt1/x@0/then    ← 栈变量 x 的 base 0 → then 分支版本
  ./tmp@0/then/else        ← 局部变量 tmp 的 base 0 → then → else 版本

解析规则:
  最后一个 @ 之前 = 变量 KV 路径
  @ 之后 = 版本号 ( / 分隔多级版本)
```

## 5. 完整编译器分析验证

### 5.1 死代码消除 (DCE)

```
a@0 + 1.0 -> x@0
if flag {
  x@0 * 2.0 -> x@0/then
  b@0 * 3.0 -> dead@0/then     ← 仅在 then 分支被定义
} else {
  x@0 * 4.0 -> x@0/else
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 3.0 -> y@0           ← 写入输出参数 y@0 = 隐式返回

编译器分析:
  1. 扫描所有读侧: x@0/merge 被使用, y@0 是输出参数 → 活
  2. dead@0/then 从未被读 → dead@0/then 是死代码 → 删除
  3. x@0/merge 是合流版本 → 其 φ 来源 x@0/then 和 x@0/else 都活
  4. 精确度等同 SSA 的 use-count DCE

结论: ✅ 完全支持
```

### 5.2 公共子表达式消除 (CSE)

```
if flag {
  a@0 + b@0 -> x@0/then        ← add(a@0, b@0)
} else {
  a@0 + b@0 -> x@0/else        ← add(a@0, b@0) — 相同操作数和操作码!
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge * 2.0 -> y@0

编译器分析:
  1. x@0/then 的定义: op=add, args=[a@0, b@0]
  2. x@0/else 的定义: op=add, args=[a@0, b@0]
  3. 操作数和操作码完全相同 → x@0/then 和 x@0/else 是同一值
  4. x@0/merge = φ(x@0/then, x@0/else) → 但两者等同 → φ 退化为 x@0/then
  5. 消除 x@0/else (else 分支直接用 x@0/then)

结论: ✅ 完全支持。版本号暴露操作数身份 → 直接比较
```

### 5.3 全局值编号 (GVN)

```
a@0 + b@0 -> t@0
c@0 + d@0 -> u@0
if flag {
  a@0 + b@0 -> x@0/then        ← VN = hash(add, a@0, b@0) = 42 (同 t@0)
} else {
  c@0 + d@0 -> x@0/else        ← VN = hash(add, c@0, d@0) = 17 (同 u@0)
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 1.0 -> y@0

编译器分析:
  分配 VN:
    t@0 = hash(add, a@0, b@0) = 42
    u@0 = hash(add, c@0, d@0) = 17
    x@0/then = 42  ← 与 t@0 相同 → 可互相替换
    x@0/else = 17  ← 与 u@0 相同 → 可互相替换
    x@0/merge = φ(42, 17) → VN = hash(φ, 42, 17) = 99
    y@0 = hash(add, 99, 1.0)

结论: ✅ 完全支持。φ 的 VN 由来源 VN 唯一确定
```

### 5.4 稀疏条件常量传播 (SCCP)

```
constant(true) -> flag@0
if flag@0 {
  constant(5.0) -> x@0/then
} else {
  constant(3.0) -> x@0/else    ← 不可达! flag=true 永不走 else
}
# 合流: x@0/merge = φ(x@0/then, x@0/else)
x@0/merge + 1.0 -> y@0

编译器分析:
  1. all values = ⊤ (top, unknown)
  2. flag@0 = true → 常量
  3. if true → then 可达到, else 不可达
  4. then: x@0/then = 5.0 → 常量
  5. else: 不可达 → x@0/else = ⊥ (从未定义)
  6. x@0/merge = φ(x@0/then=5.0, x@0/else=⊥) = 5.0
     (不可达来源被忽略)
  7. y@0 = 5.0 + 1.0 = 6.0 → 常量

结论: ✅ 完全支持。精度等同 LLVM SCCP
```

### 5.5 循环不变量外提 (LICM)

```
a@0 * b@0 -> inv@0          ← 定义在循环外
0.0 -> acc@0
for i in 0..n {
  # 循环头合流: acc@0/loop = φ(acc@0, acc@0/loop/body)
  inv@0 + acc@0/loop -> acc@0/loop/body
}

编译器分析:
  1. inv@0 定义在循环外
  2. 循环内无 inv 的新版本 (无 inv@0/loop...)
  3. → inv@0 是循环不变量
  4. 无需外提 (已在循环外)

结论: ✅ 完全支持。版本号直接暴露不变性
```

### 5.6 寄存器分配 / 干涉图

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

编译器分析:
  1. 活性区间:
     t1@0/    [1, 7]  (定义→最后使用)
     t2@0/    [2, 5]
     t3@0/then:  [4, 6]  (then 分支内)
     t3@0/else:  [5, 6]  (else 分支内)
     t3@0/merge:  [7, 7]  (合流后)
  2. 干涉:
     t1@0 与 t3@0/merge 干涉 (区间 [1,7] ∩ [7,7])
     t3@0/then 与 t3@0/else 不干涉 (不同分支，活性不重叠)
  3. 图着色: 精确分配寄存器

结论: ✅ 完全支持。版本号提供精确活性区间
```

## 6. 与纯 SSA 的架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  纯 SSA (MLIR cf dialect):                                      │
│                                                                 │
│  @0/                                                            │
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
│      a@0 + 2.0 -> x@0/then          ← base 0 → then 分支           │
│    } else {                                                      │
│      a@0 - 2.0 -> x@0/else          ← base 0 → else 分支           │
│    }                                                            │
│    x@0/merge * 3.0 -> y@0            ← x@0/merge = φ(x@0/then, x@0/else)     │
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

## 7. 实现路线

```
Phase 1: 前端版本号生成
  ┌─────────────────────────────────────────────┐
  │ 每个 slot 维护: slot → current_base          │
  │ 线形写入: 输出 "slot@N"                      │
  │ 进入分支: 追加分支标签 "slot@N/then" / "@N/else"   │
  │ 嵌套分支: 逐级追加 "/then" / "/else"              │
  │ 合流点: 自动产生 "slot@N/merge"                  │
  │                                              │
  │ 新增代码量: ~100-200 行                      │
  │ VM: 0 行 (忽略 @ 及之后全部内容)             │
  └─────────────────────────────────────────────┘

Phase 2: 编译器合流识别
  ┌─────────────────────────────────────────────┐
  │ 解析 slot@path 格式 (@ 分隔 slot 与版本)     │
  │ /merge 后缀直接识别为合流版本                    │
  │ 分支标签驱动: 前缀匹配查找 φ 来源集合        │
  │ 无需回溯控制流结构 → 版本号自描述分支来源    │
  │                                              │
  │ 新增代码量: ~150-250 行                      │
  └─────────────────────────────────────────────┘

Phase 3: 优化 Pass 全线启用
  ┌─────────────────────────────────────────────┐
  │ 内部 SSA 已就绪                               │
  │ 复用标准优化 pass (DCE, CSE, GVN, SCCP...)   │
  │ 优化结果映射回 slot → 输出优化后 IR           │
  │                                              │
  │ 优化 pass 本身: 标准实现,无需新代码           │
  └─────────────────────────────────────────────┘
```

## 8. 结论

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
│    ✅ 控制流合流: 编译器自动识别合流版本 → 构建内部 φ       │
│    ✅ 全部 pass: DCE, CSE, GVN, SCCP, LICM, 寄存器分配     │
│    ✅ 零新语法: 版本号即后缀, 无关键字, 无 block arg         │
│    ✅ VM 零变化: slot 复用保持不变, 版本号被忽略             │
│    ✅ 向后兼容: 不加版本号照常运行 (无优化)                  │
│                                                              │
│  这不是"用 ->/<- 替代 SSA"。                                 │
│  这是"把 SSA 的编译分析能力折叠进 ->/<- 的版本号系统中"。     │
│                                                              │
│  版本号 = SSA 的值身份                                      │
│  slot = 存储实体                                            │
│  合流版本号 = φ 的隐式编码                                   │
│  结构化控制流 = CFG 的确定性描述                              │
│                                                              │
│  deepxir 获得了:                                             │
│    执行层 = slot 复用的简洁                                  │
│    分析层 = SSA 的完整能力                                   │
│    转换层 = 编译器自动推导 (零人工介入)                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

> **关联文档**:
> - [spec-control-flow-v1.md](spec-control-flow-v1.md) — §4 `->`/`<-` 与 SSA 语义层面对比 + C1~C5 方案架构
> - [control-flow.md](control-flow.md) — 控制流 IR 设计（基本块模型）
> - [frontend-control-flow.md](frontend-control-flow.md) — 前端代码生成方案
