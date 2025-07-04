# mix precision

## 1. 什么是 mix precision

mix precision 是一种混合精度训练方法，它使用 16 位浮点数和 8 位整数来训练模型，从而在保持模型精度的同时，减少显存占用和计算时间。

## 2. 为什么需要 mix precision

在深度学习中，模型通常使用 32 位浮点数进行训练，这样可以确保模型的精度。但是，32 位浮点数占用的显存较大，计算时间较长。因此，为了减少显存占用和计算时间，可以使用 mix precision 训练方法。

## 3. 关于excuter的mix precision的实现

如：

matmul(A[float16],B[float16])->C[float32] //author=miaobyte id=1 create_time=1714512000 send_time=1714512000

我们在opfactory中,把实际参数用占位符替换，注册为

matmul[authora] Tensor@float16 Tensor@float16 -> Tensor@float32

如:

matmul[authora] A@float16 b@float16 -> C@float32

同样，在opfactory中，把实际参数用占位符替换，注册为

muladd[authora] Tensor@float16 Scalar@float32-> Tensor@float16










