# DeepX 算子设计原则

## 核心设计理念

### 计算与存储解耦 
    + 通过Mem类统一管理张量
    + 除了new算子，其他算子不负责tensor的创建和释放

### 通过key（string类型）向mem统一管理器，访问tensor
    + 算子的参数均为key
    + 算子内部通过key访问mem统一管理器，获取tensor
    + 算子输出结果，也通过key访问mem统一管理器，存储tensor

## 实现规范
### 命名约定
    + `算子名_数据类型`（如relu_float32）

## 算子分类

参考onnx

 
| 算子名 | 算子扼要解释 | 公式 | 融合算子及公式 |
|--------|--------------|------|----------------|
| **算术运算** |  |  |  |
| **一元运算** |  |  |  |
| Abs | 绝对值运算 | $f(x) = \|x\|$ | 无 |
| Acos | 反余弦运算 | $f(x) = \arccos(x)$ | 无 |
| Acosh | 反双曲余弦 | $f(x) = \cosh^{-1}(x)$ | 无 |
| Asin | 反正弦运算 | $f(x) = \arcsin(x)$ | 无 |
| Asinh | 反双曲正弦 | $f(x) = \sinh^{-1}(x)$ | 无 |
| Atan | 反正切运算 | $f(x) = \arctan(x)$ | 无 |
| Atanh | 反双曲正切 | $f(x) = \tanh^{-1}(x)$ | 无 |
| Ceil | 向上取整 | $f(x) = \lceil x \rceil$ | 无 |
| Cos | 余弦运算 | $f(x) = \cos(x)$ | 无 |
| Cosh | 双曲余弦 | $f(x) = \cosh(x)$ | 无 |
| Erf | 误差函数 | $f(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$ | 无 |
| Exp | 指数函数 | $f(x) = e^x$ | 无 |
| Floor | 向下取整 | $f(x) = \lfloor x \rfloor$ | 无 |
| Log | 自然对数 | $f(x) = \ln(x)$ | 无 |
| Neg | 取负运算 | $f(x) = -x$ | 无 |
| Reciprocal | 倒数运算 | $f(x) = 1/x$ | 无 |
| Sign | 符号函数 | $f(x) = \begin{cases} -1 & x < 0 \\ 0 & x = 0 \\ 1 & x > 0 \end{cases}$ | 无 |
| Sin | 正弦运算 | $f(x) = \sin(x)$ | 无 |
| Sinh | 双曲正弦 | $f(x) = \sinh(x)$ | 无 |
| Sqrt | 平方根 | $f(x) = \sqrt{x}$ | 无 |
| Tan | 正切运算 | $f(x) = \tan(x)$ | 无 |
| Tanh | 双曲正切 | $f(x) = \tanh(x)$ | 无 |
| **二元运算** |  |  |  |
| Add | 逐元素加法 | $f(a,b) = a + b$ | 可融合乘加：$a \times b + c$ |
| Div | 逐元素除法 | $f(a,b) = a / b$ | 无 |
| Mul | 逐元素乘法 | $f(a,b) = a \times b$ | 可融合乘加 |
| Pow | 幂运算 | $f(a,b) = a^b$ | 无 |
| Sub | 逐元素减法 | $f(a,b) = a - b$ | 无 |
| **比较运算** |  |  |  |
| Equal | 相等比较 | $f(a,b) = (a == b)$ | 无 |
| Greater | 大于比较 | $f(a,b) = (a > b)$ | 无 |
| GreaterOrEqual | 大于等于比较 | $f(a,b) = (a \geq b)$ | 无 |
| Less | 小于比较 | $f(a,b) = (a < b)$ | 无 |
| LessOrEqual | 小于等于比较 | $f(a,b) = (a \leq b)$ | 无 |
| Not | 逻辑非 | $f(x) = \lnot x$ | 无 |
| **逻辑运算** |  |  |  |
| And | 逻辑与 | $f(a,b) = a \land b$ | 无 |
| Or | 逻辑或 | $f(a,b) = a \lor b$ | 无 |
| Xor | 逻辑异或 | $f(a,b) = a \oplus b$ | 无 |
| BitwiseAnd | 按位与 | $f(a,b) = a \& b$ | 无 |
| BitwiseNot | 按位非 | $f(a) = \sim a$ | 无 |
| BitwiseOr | 按位或 | $f(a,b) = a \| b$ | 无 |
| BitwiseXor | 按位异或 | $f(a,b) = a \hat{} b$ | 无 |
| BitShift | 位移运算 | $f(a,b) = a \ll b$ 或 $a \gg b$ | 无 |
| **激活函数** |  |  |  |
| Elu | 指数线性单元 | $f(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$ | 无 |
| Gelu | 高斯误差线性单元 | $f(x) = x \Phi(x)$（$\Phi$为标准正态分布CDF） | 无 |
| HardSigmoid | 分段线性Sigmoid近似 | $f(x) = \max(0, \min(1, \alpha x + \beta))$ | 无 |
| HardSwish | 分段线性Swish近似 | $f(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$ | 无 |
| Hardmax | 硬性最大概率选择 | $f(x)_i = \begin{cases} 1 & \text{if } x_i = \max(x) \\ 0 & \text{otherwise} \end{cases}$ | 无 |
| LeakyRelu | 带泄露的ReLU | $f(x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}$ | 无 |
| Mish | 自正则化激活函数 | $f(x) = x \tanh(\text{softplus}(x))$ | 无 |
| PRelu | 参数化ReLU | $f(x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}$ | 无 |
| Relu | 整流线性单元 | $f(x) = \max(0, x)$ | 无 |
| Selu | 自归一化激活函数 | $f(x) = \lambda \begin{cases} x & x > 0 \\ \alpha e^x - \alpha & x \leq 0 \end{cases}$ | 无 |
| Sigmoid | S型函数 | $f(x) = \frac{1}{1 + e^{-x}}$ | 无 |
| Softmax | 归一化指数函数 | $f(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | 无 |
| Softplus | 平滑ReLU | $f(x) = \ln(1 + e^x)$ | 无 |
| Softsign | 符号函数平滑版 | $f(x) = \frac{x}{1 + \|x\|}$ | 无 |
| ThresholdedRelu | 阈值ReLU | $f(x) = \begin{cases} x & x > \theta \\ 0 & \text{otherwise} \end{cases}$ | 无 |
| **数据变换** |  |  |  |
| **形状变换** |  |  |  |
| Cast | 数据类型转换 | 无数学公式 | 无 |
| CastLike | 按目标类型转换 | 无数学公式 | 无 |
| Flatten | 展平多维数据 | 无数学公式 | 无 |
| Reshape | 改变张量形状 | 无数学公式 | 无 |
| Squeeze | 去除维度为1的轴 | 无数学公式 | 无 |
| Transpose | 转置维度顺序 | 无数学公式 | 无 |
| Unsqueeze | 增加维度为1的轴 | 无数学公式 | 无 |
| **元素选择与索引** |  |  |  |
| ArgMax | 返回最大值索引 | 无数学公式 | 无 |
| ArgMin | 