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

// ==================== 2. ir定义格式 ====================
deepxir ir_name(ro_p1:type1,ro_param2:type2,...) -> (w_p1:type3,w_p2:type4,...)
{
    // 函数体: IR操作序列
    operation_name( ro_p1,  ro_p1)-> w_p1
    operation_name( ro_p2,  ro_p2)-> w_p2
}

// ==================== 3. 具体示例 ====================

// 示例1: 符合您要求的精确约束函数
function  constrained_matmul(
     A: tensor<?x1xFloat>,      // 第一个参数: <?x1> 且元素类型为f32或f64
     B: tensor<1x?xFloat>       // 第二个参数: <1x?> 且元素类型与 A相同
) -> tensor<?x?xFloat> {        // 返回值: <?x?> 且元素类型继承自输入
    
    // 函数体 - IR操作序列
     0 = tensor.matmul( A,  B) {
        transpose_a = false,
        transpose_b = false
    } : (tensor<?x1xFloat>, tensor<1x?xFloat>) -> tensor<?x?xFloat>
    
    return  0
}

// 示例2: 更复杂的函数，包含多个操作
function  conv_relu(
     input: tensor<1x32x32x3xf32>,
     filter: tensor<3x3x3x16xf32>
) -> tensor<1x30x30x16xf32> {
    
    // 卷积操作
     conv = tensor.conv2d( input,  filter) {
        stride = [1, 1],
        padding = "valid",
        dilation = [1, 1]
    } : (tensor<1x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x30x30x16xf32>
    
    // ReLU激活
     relu = tensor.relu( conv) : (tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
    
    return  relu
}

// 示例3: 支持动态形状和类型推断
function  dynamic_operations(
     A: tensor<?x?xf32>,
     B: tensor<?x?xf32>
) -> tensor<?x?xf32> {
    
    // 元素级加法
     add = tensor.add( A,  B) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    
    // 矩阵乘法
     matmul = tensor.matmul( add,  A) : 
        (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    
    return  matmul
}

// 示例4: 带有属性约束的函数
function  batch_norm(
     input: tensor<?x?x?x?xf32>,
     scale: tensor<?xf32>,
     bias: tensor<?xf32>
) -> tensor<?x?x?x?xf32> 
attributes {
    training = true,
    epsilon = 1e-5 : f32,
    momentum = 0.9 : f32
} {
     output = tensor.batch_norm( input,  scale,  bias) {
        epsilon = 1e-5 : f32,
        training = true
    } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
    
    return  output
}

// ==================== 4. 操作签名格式 ====================
操作签名格式:
 result = operation_name(operands) {attributes} : (input_types) -> output_type

其中:
- operands: 逗号分隔的输入SSA值 ( arg1,  arg2, ...)
- attributes: 键值对属性 {key1 = value1, key2 = value2}
- input_types: 逗号分隔的输入类型
- output_type: 单个输出类型

// ==================== 5. 类型推断规则 ====================
type_rule matmul_shape_inference {
    input_shapes = [tensor<?xMxT>, tensor<?xNxT>],
    output_shape = tensor<?x?xT>,
    constraints = [
        M.dim1 == N.dim0,  // 矩阵乘法维度匹配
        T in Numeric       // 元素类型为数值类型
    ]
}

type_rule elementwise_broadcast {
    input_shapes = [tensor<A_dimsxT>, tensor<B_dimsxT>],
    output_shape = tensor<broadcast(A_dims, B_dims)xT>,
    constraints = [
        can_broadcast(A_dims, B_dims),  // 维度可广播
        T in Numeric
    ]
}