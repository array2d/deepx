# DeepX 控制流与前端代码生成 — v1 整合设计

> 整合 `control-flow.md` 与 `frontend-control-flow.md` 两份设计文档，
> 提出 6 套不同深度的重构方案，覆盖从渐进改良到完全重写的全光谱。

---

## 1. 现状总结

### 1.1 当前 deepxir 执行模型 (VM)

```
pysdk → Redis /src/func/<name>/[i] = "dxlang line"   (编译层)
        │
VM → CALL func → translate.go → eager inline
        │                        形参绑定: {A:"./a", B:"./b"}
        ▼                        坐标化: /vthread/<vtid>/[pc]/[i,j]
  /vthread/<vtid>/[0,0] = "add"                            (执行层)
  /vthread/<vtid>/[0,-1] = "./a"
  /vthread/<vtid>/[0,1] = "./c"
```

| 已有能力 | 缺失 |
|---------|------|
| `call` + eager inline + 隐式 return | 无条件跳转 (`jump`) |
| `if cond → true/false 子树` (无合并) | 循环 (`for`/`while`) |
| 原生算子求值 (18 个) | 分支合并 (`phi`) |
| op-plat 后端路由 | 基本块抽象 |
| heap-plat 生命周期 | 结构化控制流语法 |

### 1.2 当前 pysdk 代码生成模型

```
Python Tensor op → rtf_*() → DeepxIR → UDP → [外部调度器] → Redis

特点:即刻发射, 无计算图缓存, 无延迟发射, 无函数边界, 无控制流抽象
```

---

## 2. 方案总览

```
方案                              修改深度        VM 改动      前端改动       Redis schema
──────────────────────────────────────────────────────────────────────────────────
A. 渐进改良 (Incremental)          浅            中            中            微调
B. SSA 统一 IR                     深            全重写        全重写        大改
C. 二层 IR + 调度器 lowering       中            中            中            中改
D. 纯图模式 (无控制流 IR)          最浅          最小          中            不变
E. Redis 原生控制流                中            中            小            大改
F. 全编译器模型                    最深          全重写        重写          大改
```

---

## 3. 方案 A：渐进改良 (Incremental Evolution)

> 在现有基础上增加 block-based CFG + @compile 编译模式。两份设计文档 `control-flow.md`
> 和 `frontend-control-flow.md` 的合并与精炼。

### 3.1 deepxir 侧：基本块控制流

**核心变化**：将 `if`/`for`/`while` 翻译为基本块（`@0`, `@1`）+ 终止指令（`br`/`jump`）。

```
编译层 (/src/func/<name>):            执行层 (/vthread/<vtid>/<pc>/):
                                      (translate.go eager inline 后)
@0:                                   
  newtensor("f32","[4]") -> ./x       @0/0 = "newtensor", @0/0,1 = "/data/a"
  br ./cond, @1, @2                   @0 = "br", @0/-1 = "./cond", @0/-2 = "@1", @0/-3 = "@2"

@1:                                   
  add(./x, 1) -> ./y                  @1/0 = "add", @1/0,-1 = "./x", @1/0,1 = "./y"
  jump @3                             @1 = "jump", @1/-1 = "@3"

@2:
  mul(./x, -1) -> ./y                 ...
  jump @3

@3:
  deltensor(./x)                      ...
  return
```

**VM 执行循环变化**：

```go
// 旧: 线性遍历 [i,0] → [i+1,0]
// 新: block 图遍历
func Execute(vtid) {
    block := loadBlock(vtid, "@0")
    for block != nil {
        for _, inst := range block.instructions {
            dispatch(inst)
        }
        switch block.terminator.opcode {
        case "br":
            block = eval(terminator.cond) ? loadBlock(vtid, terminator.trueBlock) : loadBlock(vtid, terminator.falseBlock)
        case "jump":
            block = loadBlock(vtid, terminator.target)
        case "return":
            popFrame(); block = parentFrame.afterCall
        case "call":
            pushFrame(terminator.funcName); block = newFrame.entryBlock
        }
    }
}
```

### 3.2 pysdk 侧：Hybrid Eager+Defer

```
即刻模式 (不变):                      编译模式 (新增):
                                      @deepxir.compile
Tensor.__add__                        def f(x):
  → rtf_add()                           if x > 0:
  → DeepxIR("add", ...)                   y = branch_a(x)
  → UDP send                             else:
  ✅ 向后兼容                              y = branch_b(x)
                                        return y
                                      ↓
                                      SymbolicTensor 拦截操作
                                      IfContext 捕获分支
                                      IRBuilder 构建 CFG
                                      finalize → /src/func/f
```

**新增模块**：`compiler.py`, `ir_builder.py`, `tracer.py`, `control_flow.py`

**关键约束**：
- `SymbolicTensor` 必须与真实 `Tensor` API 完全一致（所有 `__add__`/`__matmul__`/`.relu()` 等方法）
- `@compile` 内无法调用未标注的即刻模式代码（两种模式不混用）
- 嵌套 CALL 需要编译器感知（被调函数也必须 `@compile` 或被注册到 `/src/func/`）

### 3.3 改动清单

| 组件 | 文件 | 改动 |
|------|------|------|
| VM | `ir/instruction.go` | `IsControlOp` 新增 `br`, `jump` |
| VM | `ir/instruction.go` | `Decode` 支持 `@N` block 前缀 |
| VM | `engine/engine.go` | `Execute` 改为 block-loop |
| VM | `translate/translate.go` | block-based eager inline |
| VM | `dispatch/dispatch.go` | 新增 `Br`, `Jump` handler |
| pysdk | `nn/compiler.py` | 新文件：`@compile` 装饰器 |
| pysdk | `nn/ir_builder.py` | 新文件：`IRBuilder` + `Block` |
| pysdk | `nn/tracer.py` | 新文件：`SymbolicTensor` |
| pysdk | `nn/control_flow.py` | 新文件：`IfContext`/`WhileContext` |
| Redis | schema | 执行层 key 从 `[i,j]` 迁至 `@N/i,j` |

**工作量估算**：2-3 人月
**风险**：低。保留现有架构，增量添加能力。
**收益**：获得完整动态控制流能力，97% 现有代码保持不变。

---

## 4. 方案 B：SSA 统一 IR (SSA-based Unified IR)

> 完全重新设计 deepxir 为 SSA 形式的 IR，对标 MLIR 的 `cf` + `arith` dialect。
> 放弃 Redis 路径引用（`./x`），改用虚拟寄存器编号（`%0`, `%1`）。

### 4.1 IR 设计

```
// 新 deepxir (SSA 形式)

def example(%x: tensor<f32>[4]) -> (%y: tensor<f32>[4]) {
@0:
  %0 = newtensor f32, [4]                    // 分配
  %1 = sum %x                                // reduce
  %2 = greater %1, 0                         // 比较 → bool
  br %2, @1, @2                              // 条件分支

@1:
  %3 = add %x, 1                             // true 分支
  jump @3(%3)                                // 带值跳转 (代替 phi)

@2:
  %4 = mul %x, -1                            // false 分支
  jump @3(%4)

@3(%5: tensor<f32>[4]):                      // merge, %5 = phi(%3, %4)
  deltensor %0
  return %5
}
```

**关键变化**：
- 所有值有唯一编号（`%0`, `%1` ...）
- `reads`/`writes` 数组 → SSA 操作数/结果
- block arguments 替代 phi 节点（MLIR 风格）
- 无路径绑定——值本身就是引用

### 4.2 Redis 存储格式

```
/src/func/example           → "def example(%x:tensor<f32>[4]) -> (%y:tensor<f32>[4])"
/src/func/example/@0/0      → "newtensor f32, [4]"     ; result = %0
/src/func/example/@0/1      → "sum %x"                  ; result = %1
/src/func/example/@0/2      → "greater %1, 0"           ; result = %2
/src/func/example/@0/term   → "br %2, @1, @2"

/src/func/example/@1/0      → "add %x, 1"               ; result = %3
/src/func/example/@1/term   → "jump @3(%3)"

/src/func/example/@2/0      → "mul %x, -1"              ; result = %4
/src/func/example/@2/term   → "jump @3(%4)"

/src/func/example/@3/args   → "%5: tensor<f32>[4]"      ; block argument
/src/func/example/@3/0      → "deltensor %0"
/src/func/example/@3/term   → "return %5"
```

### 4.3 执行模型

```go
// SSA 解释器
type SSAValue struct {
    ID    int
    Type  string
    Data  interface{}  // 指向 Redis key 或内存值
}

func Execute(vtid) {
    block := entryBlock(func)
    values := map[int]SSAValue{}  // 虚拟寄存器文件

    for block != nil {
        // 分解 block arguments
        for i, arg := range block.args {
            values[arg.ID] = incomingValues[i]
        }

        for _, inst := range block.instructions {
            // 解析操作数: %0 → values[0].Data
            operands := resolveSSA(inst.operands, values)
            result := dispatchSSA(inst.opcode, operands)
            values[inst.resultID] = result  // 写回寄存器文件
        }

        // 终止指令
        switch block.terminator.opcode {
        case "br":
            block = eval(values, terminator.cond) ? loadBlock(terminator.trueBlock) : loadBlock(terminator.falseBlock)
        case "jump @N(%v)":
            block = loadBlock(terminator.target)
            incomingValues = resolveSSA(terminator.args, values)
        case "return %v":
            popFrame; returnResult = resolveSSA(values, terminator.val)
        }
    }
}
```

### 4.4 pysdk 端

```python
# SSA 模式下，@compile 生成的 IR 直接使用 SSA 格式
# SymbolicTensor 的方法返回 SSA 编号而非路径名

class SymbolicTensor:
    def __init__(self, ssa_id: int, shape, dtype):
        self._ssa_id = ssa_id

    def __add__(self, other):
        result_id = builder.alloc_ssa()
        builder.emit("add", [self._ssa_id, other._ssa_id], result_id)
        return SymbolicTensor(result_id, ...)
```

### 4.5 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 学术严谨性 | ⭐⭐⭐⭐⭐ | SSA 是编译器 IR 黄金标准 |
| 与 MLIR 对齐度 | ⭐⭐⭐⭐⭐ | 直接对标，未来可互操作 |
| 实现复杂度 | ⭐ (最高) | translate.go、ir/、engine/、dispatch/ 全部重写 |
| 调试可观测性 | ⭐⭐ | SSA 编号不直观（需要符号表映射） |
| Redis 使用效率 | ⭐⭐⭐ | 仍需大量 key 存取 SSA 值 |
| 前端迁移成本 | ⭐⭐ | pysdk + Go 前端均需适配 SSA 格式 |

**工作量估算**：6-8 人月
**风险**：高。完全推倒现有 translate + dispatch 逻辑，回归测试覆盖全部算子。
**适用场景**：如果 deepxir 长期目标是成为 MLIR 兼容的方言，这是必经之路。

---

## 5. 方案 C：二层 IR + 调度器 Lowering (Two-Level IR)

> 引入调度器（Scheduler）作为编译服务，负责 lowering：
> 结构化控制流（if/for/while as 嵌套 Region）→ 平铺基本块（br/jump）。
> VM 只看到最终的基本块格式，前端只生成结构化格式。

### 5.1 架构

```
┌──────────┐    结构化 IR        ┌──────────────┐    平铺基本块       ┌─────┐
│  pysdk   │ ──────────────────→ │   Scheduler  │ ────────────────→ │  VM │
│ (@compile│   if/for/while as   │  (新增服务)   │   br/jump only    │     │
│  模式)   │   nested regions    │              │                   │     │
└──────────┘                     │  lowering:   │                   └─────┘
                                 │  region→flat  │
┌──────────┐                     │  inline CALL  │
│  Go 前端 │ ──────────────────→ │  dead-code    │
│ (图模式) │   结构化 IR         │  elimination  │
└──────────┘                     └──────────────┘
```

**设计原则**：
- 前端只生成高层结构化 IR（接近用户意图）
- Scheduler 执行 lowering pass（结构→块、优化）
- VM 只执行最基本指令（br/jump + 计算/生命周期指令）
- 每一层有独立可测试的接口

### 5.2 结构化 IR 格式（前端输出）

```json
{
  "name": "example",
  "signature": "def example(x:tensor<f32>[4]) -> (y:tensor<f32>[4])",
  "body": {
    "type": "seq",
    "ops": [
      {"op": "newtensor", "args": ["f32", "[4]"], "result": "./a"},
      {
        "type": "if",
        "cond": {"op": "greater", "args": ["./s", "0"], "result": "./cond"},
        "then": {
          "type": "seq",
          "ops": [
            {"op": "add", "args": ["./x", "1"], "result": "./y"}
          ]
        },
        "else": {
          "type": "seq",
          "ops": [
            {"op": "mul", "args": ["./x", "-1"], "result": "./y"}
          ]
        }
      },
      {"op": "deltensor", "args": ["./a"], "result": null}
    ]
  }
}
```

### 5.3 Lowering Pass 流程

```
Pass 1: Region Flattening
  结构化 if/for/while → 基本块图 (entry/true/false/merge blocks)
  
Pass 2: Block Merging
  连续无条件跳转消除: @1 jump @2, @2 jump @3 → @1 jump @3
  
Pass 3: Dead Block Elimination
  不可达 block 移除
  
Pass 4: Eager Inline (CALL 展开)
  被调用函数体展开到调用位置，形参替换
  
Pass 5: Coordinate Assignment
  基本块 → Redis key 坐标 (/vthread/<vtid>/@N/i,j)
```

### 5.4 Redis Schema（不变）

Scheduler lowering 后输出给 VM 的格式与方案 A 完全相同。VM 不需要感知 lowering 的存在。

### 5.5 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 分层清晰度 | ⭐⭐⭐⭐⭐ | 前端/Scheduler/VM 三层独立可测试 |
| 实现复杂度 | ⭐⭐⭐ | Scheduler 是新增服务，VM 改动中等 |
| 可扩展性 | ⭐⭐⭐⭐⭐ | 新的 lowering pass 独立添加 |
| 调试可观测性 | ⭐⭐⭐⭐ | 每层输出可单独检查 |
| 运维复杂度 | ⭐⭐⭐ | 新增 Scheduler 服务需要部署/监控 |

**工作量估算**：4-5 人月
**风险**：中。Scheduler 是新服务，需要定义接口协议（Redis-based? gRPC?）。
**适用场景**：追求清晰架构分层，未来有多个前端（Python+Go+C++）需要统一 lowering。

---

## 6. 方案 D：纯图模式 (Graph-only — No Control Flow in IR)

> **deepxir 不引入任何控制流原语。** 所有控制流由前端在生成 IR 之前处理。
> deepxir 保持纯粹的 DAG（有向无环图）计算语义。

### 6.1 原理

三种控制流处理策略在**前端**完成：

```
1. 结构化控制流 (if self.bias is not None):
   → 前端 if 自然处理（图构建期），不变

2. 固定次数循环 (for i in range(10)):
   → 前端展开为 10 次指令副本（类似 C++ template unrolling）

3. 动态控制流 (while loss > 1e-6):
   → 前端转换为 "control flow tensor ops"
     例如 deepx.while_loop(cond_fn, body_fn, init_tensors)
     作为特殊算子发送给 op-plat 或 VM 的专用 handler
```

### 6.2 动态控制流算子的实现

```python
# pysdk 中暴露高层 API
def training_loop(model, data, epochs):
    # 这被 @compile 展开为 while_loop 算子
    def cond(state):
        return state.epoch < epochs
    def body(state):
        loss = model(state.x)
        state = optimizer.step(state, loss)
        state.epoch += 1
        return state
    return deepx.while_loop(cond, body, init_state)
```

对应的 deepxir：
```
# while_loop 在 IR 中是一个原子算子（op-plat 负责实现）
while_loop(cond_fn_body, body_fn_body, init_state) -> final_state
```

VM 看到的是：
```
/vthread/<vtid>/[0,0] = "while_loop"
/vthread/<vtid>/[0,-1] = "/src/func/cond_fn"    # 指向函数定义
/vthread/<vtid>/[0,-2] = "/src/func/body_fn"
/vthread/<vtid>/[0,-3] = "./init_state"
/vthread/<vtid>/[0,1] = "./final_state"
```

**while_loop 算子在哪个后端执行？**
- 如果循环体全是 tensor 操作 → op-plat 可以实现为 GPU-side loop
- 如果循环体有控制逻辑 → VM 的 dispatch 实现（类似一个"微 VM"）

### 6.3 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| VM 简洁性 | ⭐⭐⭐⭐⭐ | VM 完全不感知控制流 |
| deepxir 纯粹性 | ⭐⭐⭐⭐⭐ | 纯粹的 DAG 计算图 |
| 动态控制流 | ⭐⭐ | 只能通过特殊算子，表达能力受限 |
| GPU 优化机会 | ⭐⭐⭐⭐ | 循环在 GPU 端执行，无 host-device 往返 |
| 前端复杂度 | ⭐⭐⭐ | while_loop API 需要设计 |
| 调试 | ⭐⭐ | 控制流被掩盖在算子内部 |

**工作量估算**：1-2 人月（新增 while_loop/if_else 算子 + op-plat 支持）
**风险**：低。VM 改动最小。但动态控制流的表达能力受限于算子实现。
**适用场景**：绝大多数推理 + 简单训练循环的场景。不适合需要复杂控制流的算法（如 beam search、动态路由网络）。

---

## 7. 方案 E：Redis 原生控制流 (Redis-Native Control Flow)

> 控制流表达为 Redis key 结构，不通过文本 dxlang。
> 前端直接写 Redis key，VM 直接读 Redis key 执行。

### 7.1 核心思路

放弃"文本 dxlang → 解析 → 执行"路径，改为"Redis 结构化数据 → 直接执行"路径。

```python
# pysdk 不再生成 dxlang 文本，而是直接操作 Redis
# 通过 redis-py 直接写 key

rdb = redis.Redis()

# 函数定义
rdb.set("/src/func/example", json.dumps({
    "name": "example",
    "params": [{"name": "x", "type": "tensor", "precision": "f32"}],
    "returns": [{"name": "y", "type": "tensor", "precision": "f32"}]
}))

# if block (Redis 原生嵌套)
rdb.hset("/src/func/example/blocks", "@0", json.dumps({
    "type": "entry",
    "instructions": [
        {"op": "newtensor", "reads": ["f32", "[4]"], "writes": ["./a"]},
        {"op": "greater", "reads": ["./s", "0"], "writes": ["./cond"]}
    ],
    "terminator": {
        "type": "branch",
        "cond": "./cond",
        "true_block": "@1",
        "false_block": "@2"
    }
}))

rdb.hset("/src/func/example/blocks", "@1", json.dumps({
    "type": "then",
    "instructions": [
        {"op": "add", "reads": ["./x", "1"], "writes": ["./y"]}
    ],
    "terminator": {"type": "jump", "target": "@3"}
}))
```

### 7.2 VM 执行

```go
func Execute(vtid) {
    blockID := "@0"
    for blockID != "" {
        // 直接 GET JSON，零解析成本（相比文本 parsing）
        blockJSON := rdb.HGet(ctx, "/src/func/"+funcName+"/blocks", blockID).Val()
        block := parseBlockJSON(blockJSON)  // 一次性解析

        for _, inst := range block.Instructions {
            dispatch(inst)
        }

        switch block.Terminator.Type {
        case "branch":
            cond := evalSlot(vtid, block.Terminator.Cond)
            blockID = cond ? block.Terminator.TrueBlock : block.Terminator.FalseBlock
        case "jump":
            blockID = block.Terminator.Target
        case "return":
            popFrame()
            blockID = ""
        }
    }
}
```

### 7.3 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 解析开销 | ⭐⭐⭐⭐⭐ | JSON 解析比文本解析快 |
| 类型安全 | ⭐⭐⭐⭐ | JSON 有 schema，可验证 |
| 前端复杂度 | ⭐⭐⭐⭐ | 直接操作 Redis，无中间格式 |
| 人可读性 | ⭐ | Redis 中的 JSON blob 完全不可读 |
| 调试体验 | ⭐ | 没有 .dx 文件，调试困难 |
| 跨语言兼容 | ⭐⭐⭐⭐ | JSON 天然跨语言 |
| 扩展性 | ⭐⭐⭐⭐⭐ | 新增 block type 只需新增 JSON schema |

**工作量估算**：3-4 人月
**风险**：中高。放弃文本 dxlang → 新的生态工具需要重建（语法高亮、格式化、lint）。
**适用场景**：如果 deepxir 从不需要人类直接编写，纯机器生成，这是最高效的方案。

---

## 8. 方案 F：全编译器模型 (Full Compiler Model)

> deepxir 从 **解释型** IR 变为 **编译型** IR。
> 前端生成 dxlang → 编译器编译为字节码/机器码 → VM 执行编译产物。
> Redis 回归为纯状态存储（不再存指令）。

### 8.1 架构

```
┌──────────┐    dxlang     ┌──────────────┐    bytecode     ┌─────┐
│  pysdk   │ ───────────→  │   Compiler   │ ─────────────→  │  VM │
│ (@compile│               │  (Go/Rust)   │  .dxc 文件      │     │
│  模式)   │               │              │                  │     │
└──────────┘               │ lex → parse  │                  └─────┘
                           │ → CFG → opt  │
┌──────────┐               │ → codegen    │
│  Go 前端 │ ───────────→  │              │
│ (图模式) │    dxlang     └──────────────┘
└──────────┘
```

### 8.2 字节码设计（.dxc）

```
// 指令格式: [opcode:u8] [operands...]
// 常量池 + 指令流 + 基本块偏移表

.dxc header:
  magic:     "DXC\0"
  version:   1
  num_blocks: 4
  num_constants: 3

constants:
  [0] "f32"
  [1] "[4]"
  [2] 1.0

blocks:
  @0: offset=0x0040, instructions=4
    0x10 0x00 0x01 0xFF        // NEWTENSOR const[0], const[1] → slot[0xFF]
    0x20 0xFE 0xFD              // SUM slot[0xFE] → slot[0xFD]
    0x30 0xFD 0x02 0xFC         // CMP_GT slot[0xFD], const[2] → slot[0xFC]
    0xA0 0xFC 0x01 0x02         // BR slot[0xFC], @1, @2

  @1: offset=0x0054, instructions=2
    0x40 0xFE 0x02 0xFF         // ADD slot[0xFE], const[2] → slot[0xFF]
    0xB0 0x03                   // JMP @3

  @2: offset=0x0060, instructions=2
    0x50 0xFE 0x02 0xFF         // MUL slot[0xFE], const[2] → slot[0xFF]
    0xB0 0x03                   // JMP @3

  @3: offset=0x006C, instructions=2
    0x15 0x00                   // DELTENSOR slot[0x00]
    0xC0 0xFF                   // RET slot[0xFF]
```

### 8.3 VM 执行

```go
func Execute(vtid, dxcPath string) {
    prog := LoadDXC(dxcPath)  // 一次加载字节码到内存
    
    block := prog.EntryBlock
    slots := make([]interface{}, prog.MaxSlots)  // 槽位数组（代替 Redis key）

    for block != nil {
        ip := block.StartOffset
        end := block.EndOffset
        
        for ip < end {
            opcode := prog.Code[ip]
            operands := prog.Code[ip+1 : ip+opLen(opcode)]
            dispatch(opcode, operands, slots)  // 零 Redis 访问！
            ip += 1 + len(operands)
        }
        
        switch block.Terminator.Opcode {
        case OP_BR:
            block = slots[block.Terminator.CondSlot].(bool) ? prog.Blocks[block.Terminator.TrueID] : prog.Blocks[block.Terminator.FalseID]
        case OP_JMP:
            block = prog.Blocks[block.Terminator.TargetID]
        case OP_RET:
            popFrame()
            block = parentFrame.AfterCallBlock
        }
    }
}
```

### 8.4 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 执行性能 | ⭐⭐⭐⭐⭐ | 字节码解释（甚至 JIT）远超 Redis KV 访问 |
| 指令存储成本 | ⭐⭐⭐⭐⭐ | 单个文件 vs 每条指令多个 Redis key |
| 实现复杂度 | ⭐ (最高) | 完整的编译器前后端 + 字节码解释器 |
| 调试工具 | ⭐⭐⭐⭐ | .dxc dump → 文本格式可检查 |
| 热更新 | ⭐⭐⭐ | 需要重新编译 + reload，不如 Redis 即时生效 |
| 分布式 | ⭐⭐ | .dxc 需要在多个 VM 节点上分发 |
| Redis 依赖 | ⭐⭐⭐⭐ | 仅用于状态存储（张量值），不用于指令 |

**工作量估算**：8-12 人月
**风险**：最高。完全重写执行引擎。
**适用场景**：生产级高性能部署，执行延迟要求 <1ms 的场景。

---

## 9. 六方案横向对比

### 9.1 核心维度

| 维度 | A: 渐进 | B: SSA | C: 二层 | D: 纯图 | E: Redis Native | F: 编译 |
|------|--------|--------|---------|---------|-----------------|--------|
| VM 改动量 | 中 | 全重写 | 中 | 最小 | 中 | 全重写 |
| 前端改动量 | 中 | 重写 | 中 | 小 | 小 | 重写 |
| 新增服务 | 无 | 无 | Scheduler | 无 | 无 | Compiler |
| 控制流表达力 | 完整 | 完整 | 完整 | 受限 | 完整 | 完整 |
| 执行性能 | 中 | 中 | 中 | 中 | 中 | 高 |
| Redis RT 次数/op | ~5 | ~5 | ~5 | ~5 | ~2 | 0 |
| 可观测性 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| 风险 | 低 | 高 | 中 | 低 | 中高 | 最高 |
| 人月估算 | 2-3 | 6-8 | 4-5 | 1-2 | 3-4 | 8-12 |

### 9.2 迁移路径

```
当前 ──→ A (渐进) ──→ C (二层) ──→ B (SSA) ──→ F (编译)
  │         │              │              │
  │         │              │              └── MLIR dialect 兼容
  │         │              └── 分层清晰, lowering 独立演进
  │         └── 最低风险, 最快可用
  │
  └── E (Redis Native) 可随时切入（与 A/C 并行）
  
  D (纯图) 是 A 的一个子集——如果不需要动态控制流, A 退化为 D
```

### 9.3 决策树

```
需要多久上线控制流?
  ├── < 2 个月 → 方案 A (渐进改良)
  └── 可接受 4+ 个月
      ├── 多前端 (Python+Go+未来) → 方案 C (二层 IR)
      ├── MLIR 生态对齐          → 方案 B (SSA)
      ├── 高性能低延迟           → 方案 F (编译)
      └── 纯机器生成, 不需人类可读 → 方案 E (Redis Native)

控制流需求范围?
  ├── 仅推理 (静态图)       → 方案 D (纯图) 足够
  └── 训练 + 复杂控制流     → 方案 A/B/C/E/F
```

---

## 10. 推荐路径

### 短期（本季度）：方案 A — 渐进改良

1. 引入 `jump`/`br` opcode + block-based VM 执行循环
2. 实现 `@deepxir.compile` 装饰器 (SymbolicTensor + IfContext)
3. 向后兼容：即刻模式不受影响

### 中期（下季度）：方案 C — 引入 Scheduler 服务

1. 将 `@compile` 的 CFG 构建逻辑从 pysdk 中抽出，放入独立的 Scheduler
2. Scheduler 接收结构化 IR (JSON)，执行 lowering pass
3. Go 前端也通过 Scheduler 生成控制流（不再只生成 DOT 图）
4. 获得 clean 的三层架构：前端 → Scheduler → VM

### 长期（明年）：方案 B/F — SSA / 编译

1. 根据性能需求决定：SSA 化（方案 B）或编译化（方案 F）
2. 如果 deepxir 生态成熟且需要与 MLIR 互操作 → 方案 B
3. 如果 deepxir 部署规模扩大且延迟敏感 → 方案 F

---

## 11. 附录：关键数据结构定义

### 11.1 Block (Go, 方案 A/C/E 通用)

```go
type Block struct {
    ID           string        // "@0", "@1", ...
    Instructions []Instruction // 顺序指令列表
    Terminator   Terminator    // 恰好一条终止指令
}

type Terminator struct {
    Opcode    string   // "br" | "jump" | "return" | "call"
    Reads     []string // 操作数
    TrueBlock string   // br 的 true 目标
    FalseBlock string  // br 的 false 目标
    Target    string   // jump 的目标
}
```

### 11.2 Frame (Go, 方案 A/C/E 通用)

```go
type Frame struct {
    FuncName       string
    CurrentBlockID string
    ParentFrame    *Frame
    AfterCallBlock string  // CALL 之后的下一个 block
    Vtid           string
}
```

### 11.3 SSA Instruction (Go, 方案 B 专用)

```go
type SSAInstruction struct {
    Opcode    string
    Operands  []int   // SSA ID 列表
    ResultID  int     // SSA 结果编号
}

type SSABlock struct {
    ID           string
    Args         []SSAArg      // block arguments (替代 phi)
    Instructions []SSAInstruction
    Terminator   SSATerminator
}
```

### 11.4 DXC 字节码 (Go, 方案 F 专用)

```go
const (
    OP_NEWTENSOR byte = 0x10
    OP_DELTENSOR byte = 0x15
    OP_ADD       byte = 0x40
    OP_MUL       byte = 0x50
    OP_BR        byte = 0xA0
    OP_JMP       byte = 0xB0
    OP_RET       byte = 0xC0
    // ...
)

type DXCProgram struct {
    Magic      [4]byte
    Version    uint32
    Blocks     []DXCBlock
    Constants  []interface{}
    Code       []byte  // 指令流
    MaxSlots   uint16
}
```
