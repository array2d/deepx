// ==================== 1. 类型系统 ====================
// 基础数据类型
type f16, f32, f64, bf16,bf8      // 浮点类型
type i8, i16, i32, i64, u8    // 整数类型
type bool                     // 布尔类型

//类型约束
f32|f64  //支持2种类型之一


// Tensor类型模板
type tensor<shape; elem_type>
// shape格式: dim1xdim2x...xdimN 或 ? 表示动态维度
// 示例: tensor<10x20xf32>, tensor<?x?xi32>

//动态维度的维度变量
? //任意数字
?1 //动态维度变量1
?2 //动态维度变量2，用来告诉出现?2的tensor对应维度需要保持一致

// ==================== 2. ir定义格式 ====================
deepxir ir_name(ro_p1:type1,ro_param2:type2,...) -> (w_p1:type3,w_p2:type4,...)
{
    // 函数体: IR操作序列
    operation_name( ro_p1,  ro_p1)-> w_p1
    operation_name( ro_p2,  ro_p2)-> w_p2
}
deepxir是关键词，或者我们也可以使用function,func这些传统关键字
用来定义新的ir名

deepxir的参数，遵循左读右写的规则，没有返回值
deepxir的参数类型，既包括tensor，还有list<tensor>，也包括基础类型，以及list<基础类型>

// ==================== 3.设计思考 ====================

// ==================== 4. 具体示例 ====================

// 示例1: 包含多个操作
deepxir conv_relu(input: tensor<1x32x32x3xf32>,filter: tensor<3x3x3x16xf32>) -> (out: tensor<1x30x30x16xf32>) {
    tensor.new([1 30 30 16],f32)->conv
    tensor.conv2d( input,  filter)->conv
    tensor.relu(conv)-> out
}

// 示例3: 支持动态形状和类型推断
deepxir dynamic_operations( A: tensor<?x?xf32>,B: tensor<?x?xf32>
) -> (out: tensor<?x?xf32>) {
    tensor.add( A,  B)-> %add
    tensor.matmul( %add,  A)-> out
}

// 示例4: 带有属性约束的函数
deepxir batch_norm(
     input: tensor<?x?x?x?xf32>,
     scale: tensor<?xf32>,
     bias: tensor<?xf32>
) -> (output: tensor<?x?x?x?xf32>) {
    tensor.batch_norm( input,  scale,  bias)-> output
}
