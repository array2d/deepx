# pysdk 控制流代码生成方案

> **方案分析**：[spec-control-flow-v1.md](spec-control-flow-v1.md) — C1~C5 架构对比。
> 本文分析 `front/py` 实现 + C1 关键字 IR 的具体生成设计。

## 1. 当前 front/py 架构分析

### 1.1 整体数据流

```
用户 Python 代码 (torch-like API)
        │
        ▼
  Tensor.__add__ / __matmul__ / .relu() ...
        │
        ▼
  nn.functional.leaffunc_*  →  create_A_B_tf_C 工厂函数
        │                           │
        │                   newtensor(shape) 创建输出 Tensor
        │                   rtf_mod.rtf_op(a, b, out)  发射 IR
        │                           │
        ▼                           ▼
  DeepxIR("add", [a, b], [out])  序列化
        │
        ▼
  scheduler.send(ir)  →  UDPConn.sendto("localhost:9090")
        │
        ▼
  [外部调度器]  →  Redis /src/func/  →  VM 拾取执行
```

**关键特征：即刻发射 (Eager Emission)**

每一条 Python 层的张量操作，立即生成一条 `DeepxIR` 字符串并通过 UDP 发送。
**没有图构建阶段、没有延迟发射、没有函数边界。**

### 1.2 代码生成核心：rtf 模块

`front/py/deepx/nn/functional/rtf.py` — 所有 IR 发射的汇聚点：

```python
# rtf.py — 5 种发射模板，覆盖全部当前算子

def A_B_op_C(op, a, b, out, author):          # add(A, B) -> C
    ir = DeepxIR(op, [tensor(a), tensor(b)], [tensor(out)])
    send(ir)

def A_op_C(op, a, out, author):                # relu(A) -> C
    ir = DeepxIR(op, [tensor(a)], [tensor(out)])
    send(ir)

def A_scalar_op_C(op, a, b, out, author):      # addscalar(A, 2.0) -> C
    ir = DeepxIR(op, [tensor(a), varnum(b)], [tensor(out)])
    send(ir)

def A_B_c_op_D(op, a, b, c, out, author):      # equal(A, B, ε) -> D
    ir = DeepxIR(op, [tensor(a), tensor(b), varnum(c)], [tensor(out)])
    send(ir)

def A_b1_b2_op_C(op, a, b1, b2, out, author):  # reduce(A, dims, keepdim) -> C
    ir = DeepxIR(op, [tensor(a), vector(b1), varbool(b2)], [tensor(out)])
    send(ir)
```

每种发射模板生成一条格式为 `opname (arg1, arg2) -> (ret1)  // metadata` 的字符串。

### 1.3 操作符重载层

`front/py/deepx/tensor/elementwise.py` — Tensor 方法定义：

```python
@tensor_method
def add(self, other, out='') -> Tensor:
    from deepx.nn.functional import add as add_func
    return add_func(self, other, out)  # → leaffunc → rtf → send
```

`front/py/deepx/tensor/tensor.py` — Python 操作符重载：

```python
def __add__(self, other):   return self.add(other)
def __matmul__(self, other): return self.matmul(other)
def __mul__(self, other):   return self.mul(other)
```

### 1.4 模块系统

`front/py/deepx/nn/modules/module.py` — Module 基类：

```python
class Module:
    def register_parameter(name, param):
        # 注册参数时触发 rnewtensor(param)
        from deepx.nn.functional.leaffunc_life import rnewtensor
        rnewtensor(param)

    def __call__(self, *args):
        return self.forward(*args, **kwargs)
```

`front/py/deepx/nn/modules/linear.py` — Linear 的 forward：

```python
def forward(self, input):
    y = input @ self.weight.mT        # → matmul IR emit
    if self.bias is not None:          # ← Python if，图构建期求值！
        y = y + self.bias             # → add IR emit
    return y
```

**关键发现**：`self.bias is not None` 是 Python 级判断，在**图构建期**（第一次调用 forward）就已经求值。这是"结构化控制流"——取决于模型结构而非运行时数据。

### 1.5 参数协议

`front/py/deepx/nn/deepxir.py` — `Param` 类型标记：

| Param 类别 | Python 类型 | DeepxIR 字符串形式 | 用途 |
|-----------|-----------|-------------------|------|
| `tensor:<dtype>` | Tensor | `tensor<float32>:X` | 张量操作数和返回值 |
| `var:<precision>` | int/float | `var<int32>:42` 或 `var<float32>:0.5` | 立即数标量 |
| `var:bool` | bool | `var<bool>:true` | 布尔值 |
| `var:string` | str | `var<string>:f32` | 字符串参数 |
| `vector:<precision>` | tuple | `vector<int32>:[3 4 5]` | 形状/维度参数 |
| `listtensor:<dtype>` | tuple[Tensor] | `listtensor<float32>:[A B]` | 张量列表参数 |

### 1.6 架构特点总结

```
┌──────────────────────────────────────────────────────┐
│                    即刻发射 (Eager)                    │
│                                                      │
│  Python op → rtf → DeepxIR → UDP → 调度器 → Redis     │
│                                                      │
│  ❌ 无计算图缓存         ❌ 无延迟发射                │
│  ❌ 无函数边界感知       ❌ 无控制流抽象              │
│  ❌ 无 tracer / JIT      ❌ 无 block 概念             │
└──────────────────────────────────────────────────────┘
```

---

## 2. 控制流引入后的核心矛盾

### 2.1 矛盾的根源

有了 `if`/`for`/`while` 之后，pysdk 面临一个根本问题：

```python
# 用户想表达：运行时根据 Tensor 值决定执行哪条分支
def forward(self, x):
    cond = x > 0          # ← 这是一个 Tensor<bool>，运行时才能求值
    if cond:              # ← Python if 在图构建期就求值了！
        y = self.branch_a(x)
    else:
        y = self.branch_b(x)
    return y
```

**Python 的 `if cond` 要求 `cond` 是 `bool`，但 `cond` 是 `Tensor`，Python 的 `if` 无法根据 Tensor 的运行时值分支。**

PyTorch 也面临同样问题，解决方案是 `torch.jit.script` / `torch.compile`。

### 2.2 两种控制流

| | 结构化控制流 | 动态控制流 |
|------|-----------|---------|
| 决定时机 | 图构建期（Python 运行时） | deepxir 执行期（VM 运行时） |
| 依赖数据 | Python 值（`None`/`int`/`bool`） | Tensor 值（在 Redis/gpu 上） |
| 当前支持 | ✅ 天然支持（如 `if self.bias is not None`） | ❌ 不支持 |
| 需要 IR | 不需要（Python 本身处理） | 需要 deepxir `if`/`for`/`while` |
| 示例 | 可选的 bias/activation | 循环直到收敛、动态路由 |

**设计的重点是支持动态控制流。**

---

## 3. 前端方案：Hybrid Eager+Defer + C1 关键字 IR

> 前端生成 **C1 关键字 IR**（`if`/`for`/`while` 作为一等 opcode）。
> 后端可选：VM 直接解释（C1）或 Scheduler lowering（C2）。

### 3.1 两种模式的边界

```python
# 模式 1：即刻发射（现有，不变）
x = a + b          # 立刻生成 add(a,b) -> tmp // send via UDP

# 模式 2：编译模式 → 生成 C1 关键字 IR
@deepxir.compile
def my_func(x: Tensor) -> Tensor:
    if x.sum() > 0:           # 控制流 → 编译到关键字 IR
        y = self.branch_a(x)
    else:
        y = self.branch_b(x)
    return y
```

| 特性 | 即刻发射 (Eager) | 编译模式 (@compile) |
|------|---------|---------|
| 触发方式 | 默认（任何 Tensor op） | 装饰器显式标注 |
| 发射时机 | 每个 op 立即 send | 函数调用时一次性生成 |
| 输出格式 | 单条 DeepxIR → UDP | C1 关键字 IR → Redis `/src/func/<name>` |
| 控制流 | 仅结构化（Python if） | 完整动态（if/for/while/break/continue） |
| 函数签名 | 无 | 有（`def name(params) -> (rets)`） |

## 4. C1 关键字 IR 生成设计

### 4.1 新增模块布局

```
front/py/deepx/
├── nn/
│   ├── deepxir.py                  ← 现有 IR 类（保留）
│   ├── compiler.py                 ← 新增：编译器入口 @compile
│   ├── ir_builder.py               ← 新增：IR 构建器 (DeepxFunc, Block)
│   ├── tracer.py                   ← 新增：操作拦截器 (symbolic tensor)
│   └── control_flow.py             ← 新增：控制流捕获 (If/For/While context)
└── scheduler/
    └── client/
        ├── udpconn.py              ← 现有 UDP（保留，即刻模式用）
        └── redisconn.py            ← 新增：Redis 直连（编译模式用）
```

### 4.2 编译器工作流程

```
@deepxir.compile
def my_func(x: Tensor, threshold: float) -> Tensor:
    cond = x.sum() > threshold          ← ① SymbolicTensor 拦截
    if cond:                             ← ② IfContext 捕获
        y = layer1(x)
    else:
        y = layer2(x)
    return y

↓ 装饰器编译流程：

1. 提取签名: "def my_func(x:Tensor, threshold:float) -> (y:Tensor)"
2. 运行函数体 → SymbolicTensor 拦截所有操作
3. IfContext/WhileContext 捕获控制流为嵌套结构
4. IRBuilder 收集为 C1 关键字 IR 树
5. 序列化写入 /src/func/my_func
6. LPUSH notify:vm

生成的 C1 关键字 IR:
----------------------------------------------
def my_func(x, threshold) -> (y) {
    sum(./x) -> ./s
    greater(./s, threshold) -> ./cond
    if (./cond) {
        call layer1(./x) -> ./y
    } else {
        call layer2(./x) -> ./y
    }
    return(./y)
}
----------------------------------------------
```

### 4.3 核心组件设计

#### 4.3.1 Symbolic Tensor (tracer.py)

```python
class SymbolicTensor:
    """
    编译模式下的"占位 Tensor"。
    不持有实际数据，只记录在 IR 中的符号名。
    """
    def __init__(self, name: str, shape: tuple, dtype: str):
        self._name = name       # e.g., "./x", "./sum_tmp"
        self._shape = shape
        self._dtype = dtype
        self._ir_builder = get_current_ir_builder()  # 线程局部

    def __add__(self, other) -> 'SymbolicTensor':
        # 不发射 UDP！而是在当前 block 追加指令
        out = self._ir_builder.alloc_temp()  # 分配临时变量名
        self._ir_builder.emit("add", [self, other], [out])
        return out

    def __gt__(self, other) -> 'SymbolicTensor':
        out = self._ir_builder.alloc_temp()
        self._ir_builder.emit("greater", [self, other], [out])
        return out  # ← 返回 SymbolicTensor(dtype='bool')
```

#### 4.3.2 IR Builder (ir_builder.py)

```python
class IRBuilder:
    """编译模式下收集 IR 指令的构建器"""
    def __init__(self, func_name: str):
        self.func_name = func_name
        self.signature = None
        self.blocks: list[Block] = []
        self.current_block: Block = None
        self._temp_counter = 0

    def alloc_temp(self) -> str:
        self._temp_counter += 1
        return f"./t{self._temp_counter}"

    def emit(self, opcode: str, reads: list, writes: list):
        """在当前 block 追加指令"""
        self.current_block.add_instruction(opcode, reads, writes)

    def new_block(self) -> Block:
        block = Block(len(self.blocks))
        self.blocks.append(block)
        return block

    def finalize(self) -> str:
        """生成最终的 dxlang 字符串"""
        lines = [f"def {self.func_name}{self.signature} {{"]
        for block in self.blocks:
            lines.append(f"    @{block.id}:\n")
            for inst in block.instructions:
                lines.append(f"        {inst.to_dxlang()}\n")
        lines.append("}")
        return ''.join(lines)
```

#### 4.3.3 控制流捕获 (control_flow.py)

```python
class IfContext:
    """捕获 if/else 分支为基本块"""
    def __init__(self, cond: SymbolicTensor):
        self.ir = get_current_ir_builder()
        self.cond = cond
        self.entry_block = self.ir.current_block

    def __enter__(self):
        # 创建 then block，设置结束指令为条件分支
        self.then_block = self.ir.new_block()
        self.else_block = self.ir.new_block()
        self.merge_block = self.ir.new_block()

        # 在 entry block 追加终止指令
        self.entry_block.add_terminator("br", [self.cond, self.then_block, self.else_block])

        # 切换到 then block
        self.ir.current_block = self.then_block
        return self

    def __exit__(self, ...):
        # then block 结束时追加 jump 到 merge
        self.then_block.add_terminator("jump", [self.merge_block])

        # 切换到 else block (支持 with IfContext(cond): ... else: ... 语法)
        # 否则 else block 为空 (直接 jump merge)

        # 切换到 merge block
        self.ir.current_block = self.merge_block
```

#### 4.3.4 编译器装饰器 (compiler.py)

```python
def compile(func):
    """
    将 Python 函数编译为 deepxir 函数定义，写入 Redis /src/func/<name>。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 提取签名
        sig = extract_signature(func)

        # 2. 创建 IR builder
        builder = IRBuilder(func.__name__)
        builder.signature = sig
        set_current_ir_builder(builder)

        # 3. 创建入口 block + 形参 SymbolicTensor
        builder.current_block = builder.new_block()
        params = create_symbolic_params(sig, args)

        # 4. 执行函数体（操作被拦截）
        result = func(*params, **kwargs)

        # 5. 在末尾 block 追加隐式 return
        builder.current_block.add_terminator("return", [result])

        # 6. 生成 dxlang + 写入 Redis
        dxlang = builder.finalize()
        rdb.set(f"/src/func/{func.__name__}", sig)
        for i, block in enumerate(builder.blocks):
            for j, inst in enumerate(block.instructions):
                rdb.set(f"/src/func/{func.__name__}/{i}/{j}", inst.to_dxlang())

        # 7. 通知 VM
        rdb.lpush("notify:vm", f"func:{func.__name__}")

        # 8. 清理线程局部
        clear_current_ir_builder()

    return wrapper
```

### 4.4 支持的控制流语法

```python
@deepxir.compile
def example(x: Tensor, n: int) -> Tensor:
    # === if/else ===
    if x.sum() > 0:                   # 条件为 SymbolicTensor
        y = layer_a(x)
    else:
        y = layer_b(x)

    # === for (固定次数) ===
    for i in range(10):               # Python range → 编译期展开为顺序指令序列
        y = y + x                     #    （不需要 deepxir for 控制流）

    # === for (动态边界) ===
    for _ in deepxir.loop(cond=lambda: y.sum() < threshold):
        y = y * 0.9 + x * 0.1

    # === while ===
    while y.sum() > 1e-6:             # 条件为 SymbolicTensor
        y = y * 0.5

    return y
```

### 4.5 与现有即刻模式的共存

```
┌──────────────────────────────────────────────────────────────┐
│                      pysdk 入口                               │
│                                                              │
│  user_code.py                                                │
│    │                                                         │
│    ├── x = a + b          → Tensor.__add__                   │
│    │                         → rtf_add() → DeepxIR → UDP     │
│    │                         ✅ 即刻发射（现有，不变）         │
│    │                                                         │
│    └── @compile def f(x):  → compiler.py                     │
│          if x > 0:           → IfContext 捕获                │
│              ...             → SymbolicTensor 拦截            │
│          return y            → finalize → dxlang → Redis     │
│                              ✅ 编译模式（新增）              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Redis 存储格式（C1 关键字 IR）

编译器输出的 C1 关键字 IR 在 Redis 中以嵌套 key 表达：

```
C1 关键字 IR:
----------------------------------------------
def example(A:int, B:int) -> (C:int) {
    newtensor("f32", "[4]") -> ./x
    sum(./x) -> ./s
    greater(./s, 0) -> ./cond

    if (./cond) {
        add(./x, 1) -> ./y
    } else {
        mul(./x, -1) -> ./y
    }

    deltensor(./x)
    return(./y)
}
----------------------------------------------

对应的 Redis key（嵌套同构）:

/src/func/example              = "def example(A:int, B:int) -> (C:int)"
/src/func/example/0            = "newtensor(\"f32\", \"[4]\") -> ./x"
/src/func/example/1            = "sum(./x) -> ./s"
/src/func/example/2            = "greater(./s, 0) -> ./cond"
/src/func/example/3            = "if"
/src/func/example/3/cond       = "./cond"
/src/func/example/3/then/0     = "add(./x, 1) -> ./y"
/src/func/example/3/else/0     = "mul(./x, -1) -> ./y"
/src/func/example/4            = "deltensor(./x)"
/src/func/example/5            = "return(./y)"
```

**关键属性**：IR 的嵌套结构与 Redis key 的嵌套结构完全同构。
VM 在执行 `if`/`for`/`while` 时自然地按照 key 子树进入子作用域。

---

## 6. 与 VM 翻译层的配合

当前 VM 的 `translate.go` 处理 CALL 指令时：
```
CALL func_name(args...) →
  1. 读取签名: GET /src/func/<name>
  2. MGET 函数体 (按数字后缀排序)
  3. 逐条翻译为 vthread 执行层坐标
  4. 追加隐式 return
```

引入 block 格式后，`translate.go` 需要升级：

```
CALL func_name(args...) →
  1. 读取签名: GET /src/func/<name>
  2. 读取 blocks: KEYS /src/func/<name>/@*
  3. 为每个 block 读取指令序列 (按数字后缀排序)
  4. 构建 CFG: 解析每个 block 的终止指令 (br/jump/return)
  5. Eager inline: 形参替换 + 写入 vthread 子树
  6. Block 终止指令保留为控制流 opcode (br/jump/return)
```

**关键变化**：之前的翻译是线性的（指令列表），现在是图结构的（block 列表 + 跳转边）。

---

## 7. 与 Go 前端的对比

Go 前端 (`front/go/deepx/`) 采用 **图模式 (Graph Mode)**：

```go
func (m *Transformer) Forward(x *Tensor) *Tensor {
    for _, layer := range m.layers {       // ← Go for, 图构建期展开
        x = layer.Forward(x)
    }
    return x
}
```

Go 的 `for` 循环在编译期展开（图构建期），生成的是**静态计算图**。Go 前端未来也需要支持动态控制流——两种前端可以共享同一套编译期抽象（IR Builder / Block / CFG）。

| 维度 | Python 即刻模式 | Python 编译模式 | Go 图模式 |
|------|---------|---------|--------|
| 发射时机 | 每个 op 立即 | 函数调用时批量 | 图构建完成后 |
| 控制流 | Python if/for（构建期） | deepxir if/for/while（执行期） | Go if/for（构建期） |
| 输出格式 | 单条 IR → UDP | 完整 dxlang → Redis | DOT 图 → 文件 |
| 嵌套调用 | 无 | 有（装饰器标注） | 有（直接调用） |
| 动态控制流 | ❌ | ✅ | ❌（未来需要） |

---

## 8. 实现路线图

### Phase 1: IR Builder + SymbolicTensor（线性函数）

1. 实现 `SymbolicTensor`（`__add__`/`__mul__` 等拦截，生成 C1 IR）
2. 实现 `IRBuilder`（单 scope 指令收集）
3. 实现 `@deepxir.compile` 装饰器（无分支，纯线性）
4. Redis 写入 (`/src/func/<name>`)
5. VM `translate.go` 支持 C1 嵌套 key 格式

### Phase 2: 控制流捕获（C1 关键字）

1. 实现 `IfContext` → 生成 `if (cond) { ... } else { ... }` 关键字 IR
2. 实现 `WhileContext` → 生成 `while (cond) { ... }`
3. 实现 `ForContext` → 生成 `for (init; cond; step) { ... }`
4. 实现 `break`/`continue` 上下文感知
5. VM 实现嵌套作用域栈，直接解释关键字

### Phase 3: 基本块执行（为 C2/C5 准备）

1. VM 新增 block-loop 执行模式
2. 关键字 → 基本块 lowering 模块（先放在 VM 内）
3. `jump`/`br` 终止指令

### Phase 4: 优化与生产化

1. Scheduler 服务（lowering 从 VM 抽出）
2. 基本块合并、死代码消除
3. Eager 与 @compile 混用

---

## 9. 关键设计决策

### 为什么不用 TorchScript 的 Script 模式

TorchScript 需要维护一个完整的 Python 子集编译器（类型推断、控制流分析、字节码拦截），复杂度极高。deepxir 只需要支持张量运算 + 基本控制流，拦截 `Tensor` 操作 + 控制流关键字就足够了。

### 为什么保留即刻模式

即刻模式（`rtf → DeepxIR → UDP`）是调试、探索式开发、单步测试的基础。编译模式是性能优化和复杂控制流的手段。两种模式互补而非替代。

### 为什么函数是编译单元

以**函数**为编译单元而不是整个程序，因为：
1. deepxir 的函数就是 VM 的调用/调度单元
2. 编译一个函数 → 写入 `/src/func/<name>` → VM 的 `CALL` 可直接使用
3. 与现有的 `testdata/*.dx` 文件格式兼容
4. 与 Go 前端的模块/层概念对齐
