# DeepX IR (Intermediate Representation) 格式规范

DeepX IR 采用简洁的文本格式来表示张量运算。主要分为函数定义(funcdef)和函数调用(funccall)两种模式。

## 基本语法规则

1. 使用 `->` 分隔输入参数和返回值
2. 参数之间使用逗号(,)分隔
3. 向量类型的值使用空格分隔元素
4. 参数和返回值可选择性地用括号()包裹
5. 可在指令后添加元数据，使用 `//` 分隔

## 函数调用(funccall)模式

函数调用模式用于实际执行操作，语法更简洁。

示例:
matmul A,B -> C
sum(A,[1 2 3]) -> B
newtensor 3 4 5 -> T1

## 函数定义(funcdef)

函数定义由excuter层负责注册实现,用于声明操作的参数和返回值类型。excuter通过注册funcdef来声明其支持的tensorfunc。

因此需要设置参数、返回值的详细类型约束

语法示例:
```
matmul(Tensor<float32|float64> A, Tensor<float32|float64> B) -> Tensor<float32|float64> C
sum(Tensor<any> A, vector<int32> dim) -> Tensor<any> B
newtensor(vector<int32> shape) -> Tensor<float32> T1
```

## 元数据格式

可在指令后添加元数据信息:

```
matmul(A,B)->C //id=1 created_at=123456789 sent_at=123456790
```

支持的元数据字段:
- id: 操作ID
- author: 作者，部分tensorfunc的实现，如matmul，会有多实现，需要指定作者以根据环境指定最优实现
- created_at: 创建时间戳
- sent_at: 发送时间戳

## 类型系统

对于tensorfunc的类型系统，我们只关心与tensor相关的类型系统

参考 excuter/common/src/deepx/dtype.hpp

```
{
    类型: 
         var
         vector
         tensor
         listtensor
    精度:
         float64
         float32
         float16
         bfloat16
         fp8
         fp4
         int64
         int32
         int16
         int8
         int4
         string//可以用来引用其他var或tensor的name
}
```
多精度支持可以用|分隔,如float32|float64


## funcdef

excuter 负责定义其支持的tensorfunc

1. 矩阵乘法:
```
# funcdef
matmul(Tensor<float32|float64> A, Tensor<float32|float64> B) -> Tensor<float32|float64> C

# funccall  
matmul A,B -> C
// rtf(remote tensor func)解析器会自动解析参数和返回值的列表
// excuter会从mem获取A，B，C这3个tensor，并执行matmul操作
```

2. 张量求和:
```
# funcdef
sum(Tensor<any> input, vector<int32> dims,var<bool> keepdim) -> Tensor<any> output

# funccall
sum(T1,[0 1],true) -> T2
// rtf(remote tensor func)解析器会自动解析参数和返回值的列表
// 其中[0 1]会被解析为vector<int32>，便于excuter执行时使用
// true会被解析为var<bool> keepdim，便于excuter执行时使用
// excuter会从mem获取T1，T2这2个tensor，并执行sum操作
```

3. 创建新张量:
```
# funcdef
newtensor(vector<int32> shape) -> Tensor<float32> output

# funccall
newtensor 3 4 5 -> T1
```