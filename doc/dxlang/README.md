# dxlang

> dxlang 是 deepx 元程级的编程语言。**deepxir 即 dxlang 的指令设计部分**——两者不是两层，而是同一语言的不同视角。

## 0. 统一语法哲学

LLVM 将表示划分为多层：高级语言 → 中层 IR → 低级机器码，各层语法互不相通，编译器在其中逐层翻译。

dxlang 不这样做。**dxlang 以同一种语法，同时承担四种职能**：

| 职能 | 说明 |
|------|------|
| **VM 指令执行** | `add(A, B) -> C` 即 VM 可逐条解码执行的操作码 |
| **高级语言定义** | `def gemm(...) -> (...) { ... }` 即程序员编写的函数 |
| **编译器分析** | `->`/`<-` 数据流箭头天然可做 use-def 链分析，无需构造 SSA |
| **人类可阅读** | 纯文本、无编号寄存器、无 phi 节点，源码即 IR |

**与 LLVM 的本质区别**：

```
LLVM:  C++/Rust  →  LLVM IR  →  Machine Code
       三层语法，逐层翻译，互不兼容

dxlang:  def  →  op(A,B)->C  →  /vthread/[i,j]
        同一语法，同一表示，视角不同而已
```

**deepxir** 即这条指令 `op(A, B) -> C` 的设计——它不是独立的一层，它就是 dxlang 本身，是 dxlang 在"指令"视角下的名字。

### 执行模型定位

**底座是分布式的，语言是单线程的。**

deepx 的底层是一个分布式系统：Redis KV 空间、heap-plat 跨进程管理 shm、op-plat 常驻消费计算指令、VM 多 worker 并行调度 vthread——但 dxlang **使用起来就像 SQL 一样**，程序员看到的是极其简单的单线程顺序逻辑：

```dxlang
# 程序员视角：就几条顺序语句，像写 SQL 一样简单
def hadamard3() -> ("/data/result") {
    newtensor("f32", "[128]") -> "/data/a"
    newtensor("f32", "[128]") -> "/data/b"
    mul("/data/a", "/data/b") -> "/data/temp"    # GPU 执行，程序员无感
    deltensor("/data/a")
}
```

| 底座（分布式） | 语言（单线程） |
|----------------|----------------|
| Redis KV 空间跨进程共享 | 程序员只写 `A + B -> C` |
| heap-plat 管理 shm 生命周期 | `newtensor` / `deltensor` 就像 `malloc` / `free` |
| op-plat 被动消费 GPU 指令 | `mul` / `add` 不感知 Metal / CUDA |
| VM 多 worker 并行调度 | `def` 函数定义，调用即 `CALL` 翻译 |

这就是 dxlang 的核心设计意图：**为 AI 任务服务的声明式语言**——数据流用箭头 (`->`) 表达，计算交给平台代理，使用者只需关心"什么算子、什么数据、什么顺序"，无需关心 GPU 型号、内存布局、进程间通信。

与 C/CUDA 的本质差异：

```
C/CUDA:  程序员管理一切 ── 内存分配、设备选择、数据传输、kernel launch
dxlang:  程序员只写数据流 ── 底座自动调度、分配、执行、回收
```

dxlang 不接触物理地址，不管理内存分配，不感知设备拓扑——这些全部由 heap-plat 和 op-plat 的常驻进程代理。

## 1. 类型系统

### 基础数据类型
```
type f16, f32, f64, bf16, bf8
type i8, i16, i32, i64, u8
type bool
type string
type tensor
```

### 类型约束
```
f32|f64
```

### Tensor 类型模板
```
type tensor<shape, elem_type>
```
- shape 格式：dim1xdim2x...xdimN，或使用 `?` 表示动态维度  
- 示例：`tensor<10x20xf32>`, `tensor<?x?xi32>`

tensor 也可以没有 shape 和 dtype 约束：
```
func addscalar(A:tensor, b:i8|i16|i32|i64) -> (c:tensor) { ... }
```

### 动态维度变量
- `?` 任意数字  
- `?1` 动态维度变量 1  
- `?2` 动态维度变量 2  
- 示例：`tensor<?1x?2xf32>`


### 数组类型
```
type[]
```
list 可以与基础类型与 tensor 组合
## 2.控制流

> **v1 整合设计**：[spec-control-flow-v1.md](spec-control-flow-v1.md) — 6 套重构方案全光谱对比 (渐进 → SSA → 编译) + SSA vs `->`/`<-` 语义对比。
> 详细方案见 [control-flow.md](./control-flow.md)、[frontend-control-flow.md](./frontend-control-flow.md)。
> 编译器分析：[compiler-analysis-ssa-vs-arrow.md](./compiler-analysis-ssa-vs-arrow.md) — `->`/`<-` 能否替代 SSA 做编译分析 (§1-8) + 变量版本号方案 (§9) + **`resolve` 创新方案填补 φ 缺口 (§10)**。

dxlang 支持分支与循环，控制流以“语义块”表达，执行时由解释器按块索引跳转。

### 分支
```
if (cond:bool) {
    op(a)-> b
    op2(b)->c
} else {
    op3(a)->b
    op4(a)->d
}
```

### 循环（迭代器）
```
var list <-[1,2,3,4,5,6,7]
for (i:i32 in range ) {
    op(i, a)-> b
}
```

## 3.函数语法

### 函数定义
```
func ir_name(ro_p1:type1, ro_p2:type2, ...) -> (w_p1:type3, w_p2:type4, ...)
{
    operation_name(ro_p1, ro_p2) -> w_p1
    operation_name(ro_p2, ro_p2) -> w_p2
}
```

### 只读/写入参数标记
dxlang 支持`<-` 与 `->`（或者`<=` 与 `=>`），用于显式区分只读与写入参数，箭头指向写入参数列表；

示例（标准格式）：
```
func gemm(A:tensor<f32>, B:tensor<f32>, alpha:f32, beta:f32, C:tensor<f32>) -> (Y:tensor<f32>) {
    matmul(A, B) -> Y
    mul(Y, alpha) -> Y
    mul(C, beta) -> C
    add(Y, C) -> Y
}
```

示例：
```
func ffn(A:tensor<?1x?2xf32>, W1:tensor<?2x?3xf32>, b1:tensor<?3xf32>, W2:tensor<?3x?4xf32>, b2:tensor<?4xf32>) -> (Y:tensor<?1x?4xf32>) {
    matmul(A, W1) => Y
    add(Y, b1) => Y
    gelu(Y) => Y
    matmul(Y, W2) => Y
    add(Y, b2) => Y
}
```

### 创建变量
```
var a<=false
var b,c<=1,2
```
- `var` 定义新对象  
- 自动类型推断：`b,c` 推断为 `int` 

## 4. KV空间与执行模型

### KV空间组织

dxlang采用kv地址系统，从而实现跨节点的统一地址，而非传统的单机进程内存模型


简单的串行函数与语句可由 python-sdk 直接写入 KV（如 Redis），示例结构：
```
/func/gemm = (A:tensor<f32>, B:tensor<f32>, alpha:f32, beta:f32, C:tensor<f32>) -> (Y:tensor<f32>)
/func/gemm/0 = matmul(A, B) -> Y
/func/gemm/1 = mul(Y, alpha) -> Y
/func/gemm/2 = mul(C, beta) -> C
/func/gemm/3 = add(Y, C) -> Y
```

控制流 if：
```
.../0 = v
.../1 = if(cond)
.../1/true/0 = add(A, B) -> Y
.../1/false/0 = sub(A, B) -> Y
```


控制流for


### 函数体的复杂控制流索引
对于包含控制流的函数体




## 5. pysdk 代码生成

> 详细设计方案见 [frontend-control-flow.md](./frontend-control-flow.md) —— 当前 front/py 架构分析 + 三种方案对比 + Hybrid Eager+Defer 推荐方案 + 实现路线图。

核心结论：保留现有即刻发射模式不变；通过 `@deepxir.compile` 装饰器新增编译模式，支持 `if`/`for`/`while` 动态控制流的 deepxir 代码生成。

## 6. 设计思考
dxlang 采用简洁文本格式表达类型约束、运算定义与运算体，便于阅读与解析。  
dxlang 不是 SSA，调用时遵循一侧读、另一侧写的规则，参数列表支持多个。  
dxlang 作为调度语义与协议载体，不负责算子实现与存储生命周期。  

## 7. 具体示例

### 示例 1：融合 Linear + 归一化
```
func fused_linear_norm(
    A: tensor<?1x?2xf32>,
    W: tensor<?2x?3xf32>,
    b: tensor<?3xf32>,
    axis: i32,
    keepdims: bool
) -> (out: tensor<?1x?3xf32>) {
    newtensor(?1x?3, f32)->(mm)
    matmul(A, W)-> (mm)
    newtensor(?1x?3, f32)-> bias
    add(mm, b)-> bias
    deltensor(mm)-> mm
    newtensor(?1, f32)-> mean
    sum(bias, axis, keepdims)-> mean
    newtensor(?1x?3, f32)-> centered
    sub(bias, mean)-> centered
    deltensor(bias)-> bias
    deltensor(mean)-> mean
    newtensor(?1x?3, f32)-> sq
    mul(centered, centered)-> sq
    deltensor(centered)-> centered
    newtensor(?1, f32)-> var
    sum(sq, axis, keepdims)-> var
    deltensor(sq)-> sq
    constant(1e-5)-> eps
    newtensor(?1, f32)-> var_eps
    add(var, eps)-> var_eps
    deltensor(var)-> var
    deltensor(eps)-> eps
    newtensor(?1, f32)-> std
    sqrt(var_eps)-> std
    deltensor(var_eps)-> var_eps
    div(std, std)-> std
    deltensor(std)-> std
    div(centered, std)-> out
}
```

```
func example_use_fused_linear_norm() -> (out: tensor<2x3xf32>) {
    newtensor([2,4], f32)-> A
    newtensor([4,3], f32)-> W
    newtensor([3], f32)-> b
    fused_linear_norm(A, W, b, 1, false) -> out
}
```

### 示例 2：融合 Attention score + Softmax
```
func fused_attention_scores(
    Q: tensor<?x?xf32>,
    K: tensor<?x?xf32>,
    axis: list<i32>,
    keepdims: bool,
    shape_scores: list<i32>,
    shape_sum: list<i32>
) -> (out: tensor<?x?xf32>) {
    newtensor(shape_scores, f32)-> scores_tmp
    matmul(Q, K)-> scores_tmp
    newtensor(shape_scores, f32)-> exp_tmp
    exp(scores_tmp)-> exp_tmp
    deltensor(scores_tmp)-> scores_tmp
    newtensor(shape_sum, f32)-> sum_tmp
    sum(exp_tmp, axis, keepdims)-> sum_tmp
    div(exp_tmp, sum_tmp)-> out
    deltensor(exp_tmp)-> exp_tmp
    deltensor(sum_tmp)-> sum_tmp
}
```