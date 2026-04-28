+ 向量X的最大值
返回向量中绝对值最大的元素的索引
```
cublasI[s|d|c|z]amax(cublasHandle_t handle, int n, const T *x, int incx, T *result)
```

+ 向量X的最小值
```
cublasI[s|d|c|z]amin(cublasHandle_t handle, int n, const T *x, int incx, T *result)
```

+ 向量X的绝对值之和
```
cublasI[s|d|c|z]asum(cublasHandle_t handle, int n, const T *x, int incx, T *result) 
```

+ 向量X的标量乘法
 
result = alpha * x + y
 
```
cublasI[s|d|c|z]axpy(cublasHandle_t handle, int n, 
                    const T *alpha,
                    const T *x, int incx,
                    T *y, int incy)
```
 
 
+ 向量X的copy

y = x

```
cublasI[s|d|c|z]copy(cublasHandle_t handle, int n, const T *x, int incx, T *y, int incy)
```


+ 向量X的dot

?

```
cublasI[s|d|c|z]dot(cublasHandle_t handle, int n, 
                    const T *x, int incx, 
                    const T *y, int incy, 
                    T *result)
```

+ 向量X的nrm2 

计算向量x的欧几里得范数

```
cublasI[s|d|c|z]nrm2(cublasHandle_t handle, int n, const T *x, int incx, T *result)
``` 

+ 向量X的rot()
在x， y平面上按cos（alpha）=c，sin（alpha）=s定义的角度逆时针旋转


```
cublasI[s|d|c|z]rot(cublasHandle_t handle, int n,
                     T *x, int incx, 
                     T *y, int incy,
                     const T *c,const T *s)
```


## Level-3 BLAS操作

列主格式存储

+ gemm (通用矩阵乘法)
计算 C = α * op(A) * op(B) + β * C
其中op(X)可以是X或X^T

```
cublasI[s|d|c|z]gemm(cublasHandle_t handle,
                     cublasOperation_t transa, cublasOperation_t transb,
                     int m, int n, int k,
                     const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc)
```

+ gemmBatched (批量矩阵乘法)
同时计算多个独立的矩阵乘法操作
数学公式: C[i] = α * op(A[i]) * op(B[i]) + β * C[i], i ∈ [0,batchCount)

+ gemmStridedBatched (步进批量矩阵乘法)
处理内存连续的批量矩阵乘法，通过stride指定每个矩阵的步长
数学公式: 同gemmBatched，但矩阵在内存中以固定步长排列

+ gemmGroupedBatched (分组批量矩阵乘法)
按组处理批量矩阵乘法，每组可以有不同的维度参数
数学公式: 同gemmBatched，但可以按组设置不同的m,n,k

+ geam (矩阵加法与转置)
计算 C = α * op(A) + β * op(B)
其中op(X)可以是X或X^T
可用于实现矩阵转置、加法、缩放等操作

+ dgmm (对角矩阵乘法)
计算对角矩阵与普通矩阵的乘法
数学公式:
- 左乘模式: C = diag(x) * A
- 右乘模式: C = A * diag(x)

+ gemmEx (扩展精度矩阵乘法)
支持混合精度计算，如:
- FP16输入，FP32累加和输出
- INT8输入，INT32累加，FP32输出
数学公式: 同gemm，但支持不同数据类型

+ GemmBatchedEx (扩展精度批量矩阵乘法)
批量版本的gemmEx，支持混合精度

+ cublasGemmStridedBatchedEx (扩展精度步进批量矩阵乘法)
步进批量版本的gemmEx，支持混合精度

+ cublasGemmGroupedBatchedEx (扩展精度分组批量矩阵乘法)
分组批量版本的gemmEx，支持混合精度

+ Csyrk3mEx (对称矩阵更新)
计算对称矩阵的rank-k更新，使用3M算法优化复数运算
数学公式: C = α * op(A) * op(A)^T + β * C
其中C为对称矩阵
