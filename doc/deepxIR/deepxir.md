# DeepX IR（deepxir）规范

## 1. 类型系统

### 基础数据类型
```
type f16, f32, f64, bf16, bf8    // 浮点类型
type i8, i16, i32, i64, u8       // 整数类型
type bool                       // 布尔类型
```

### 动态长度类型
```
list<type>   // list 可以和以上基础类型组合
```

### 类型约束
```
f32|f64   // 支持两种/多种 类型之一
```

### Tensor 类型模板
```
type tensor<shape, elem_type>
```
- shape 格式：dim1xdim2x...xdimN，或使用 `?` 表示动态维度。 最后一个x后的是精度。 
- 示例：`tensor<10x20xf32>`, `tensor<?x?xi32>`

tensor 也可以没有 shape 和 dtype 的约束，例如：
```
deepxir addscalar(A:tensor, b:i8|i16|i32|i64) -> (c:tensor) { ... }
```
表示任意 shape、任意 dtype 的 tensor 都可作为参数。

### 动态维度变量
- `?` 任意数字  
- `?1` 动态维度变量 1  
- `?2` 动态维度变量 2（用于表示同名变量处维度需一致）  
- 示例：`tensor<?1x?2xf32>`

## 2. IR 定义格式

语法示例：
```
deepxir ir_name(ro_p1:type1, ro_param2:type2, ...) -> (w_p1:type3, w_p2:type4, ...)
{
    // 函数体：IR 操作序列
    operation_name(ro_p1, ro_p2) -> w_p1
    operation_name(ro_p2, ro_p2) -> w_p2
}
```
- `deepxir` 为关键字，也可使用 `function`、`func`、`def` 等,作用完全相同的。  
- 参数遵循“左读右写”规则（无返回值；通过写入参数实现输出）。  
- 参数类型支持：`tensor`、`list<tensor>`、基础类型，以及基础类型的 list。

## 3. 设计思考
DeepX IR 采用简洁的文本格式表示张量类型约束、运算定义与运算体，便于阅读与解析。
deepx不是ssa，调用时，依然遵循左读右写的参数列表原则，右写的参数列表支持多个。

## 4. 具体示例

### 示例 1：融合 Linear + 归一化
```
deepxir fused_linear_norm(
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

下面给出一个完整的 `deepxir` 调用示例：在一个 IR 中先构造输入张量和辅助参数，然后调用 `fused_linear_norm`，输出 `out`。

```
deepxir example_use_fused_linear_norm() -> (out: tensor<2x3xf32>) {
    newtensor([2,4], f32)-> A
    newtensor([4,3], f32)-> W
    newtensor([3], f32)-> b
    fused_linear_norm(A, W, b, 1, false) -> out
}
```

该示例展示了如何在 IR 中构造必要的张量/参数并调用 `fused_linear_norm`，其中 `out` 的类型为 `tensor<2x3xf32>`，与 `W` 的列数和 `A` 的行数对应。

### 示例 2：融合 Attention score + Softmax
```
deepxir fused_attention_scores(
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