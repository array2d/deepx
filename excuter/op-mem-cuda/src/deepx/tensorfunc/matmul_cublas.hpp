#ifndef DEEPX_TENSORFUNC_MATMUL_CUBLAS_HPP
#define DEEPX_TENSORFUNC_MATMUL_CUBLAS_HPP

#include "cublas_v2.h"
#include <cuda_fp16.h> // 为了支持half精度
#include <cuda_bf16.h>
#include "deepx/tensor.hpp"

#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/cuda.hpp"

namespace deepx::tensorfunc
{
    using namespace deepx;
    template <>
    struct matmulDispatcher<cublas, float>
    {
        static void matmul(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
        {
            if (!check_matmul_shape(A.shape, B.shape))
            {
                throw std::invalid_argument("A.shape could not matmul with B.shape");
            }

            static CublasHandle handle;
            const float alpha = 1.0f;
            const float beta = 0.0f;

            int64_t batch_size = A.shape.size / (A.shape[-2] * A.shape[-1]);

            // 获取矩阵维度
            int m = A.shape[-2];
            int k = A.shape[-1];
            int n = B.shape[-1];

            // 计算步长(stride) - 对行主序
            int64_t stride_a = m * k;
            int64_t stride_b = k * n;
            int64_t stride_c = m * n;

            if (batch_size > 1)
            {
                // 使用批量版本，注意: 为处理行主序，交换了顺序并转置操作
                auto status = cublasSgemmStridedBatched(handle.get(),
                                                        CUBLAS_OP_N,  // B不转置
                                                        CUBLAS_OP_N,  // A不转置
                                                        n, m, k,      // 交换m,n处理行主序
                                                        &alpha,
                                                        B.data, n, stride_b,  // B在前
                                                        A.data, k, stride_a,  // A在后
                                                        &beta,
                                                        C.data, n, stride_c,  // 输出维度对应调整
                                                        batch_size);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cublasSgemmStridedBatched failed");
                }
            }
            else
            {
                // 单个矩阵乘法，同样交换顺序处理行主序
                auto status = cublasSgemm(handle.get(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        n, m, k,    // 交换m,n
                                        &alpha,
                                        B.data, n,  // B在前
                                        A.data, k,  // A在后
                                        &beta,
                                        C.data, n); // 调整leading dimension

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cublasSgemm failed");
                }
            }
        }
    };
    template <>
    struct matmulDispatcher<cublas, double>
    {
        static void matmul(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
        {
            if (!check_matmul_shape(A.shape, B.shape))
            {
                throw std::invalid_argument("A.shape could not matmul with B.shape");
            }

            static CublasHandle handle;
            const double alpha = 1.0;
            const double beta = 0.0;

            // 获取批次数量
            int64_t batch_size = A.shape.size / (A.shape[-2] * A.shape[-1]);

            // 获取矩阵维度
            int m = A.shape[-2];
            int k = A.shape[-1];
            int n = B.shape[-1];

            // 计算步长
            int64_t stride_a = m * k;
            int64_t stride_b = k * n;
            int64_t stride_c = m * n;

            if (batch_size > 1)
            {
                auto status = cublasDgemmStridedBatched(handle.get(),
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        m, n, k,
                                                        &alpha,
                                                        A.data, m, stride_a,
                                                        B.data, k, stride_b,
                                                        &beta,
                                                        C.data, m, stride_c,
                                                        batch_size);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cublasDgemmStridedBatched failed");
                }
            }
            else
            {
                // 单个矩阵乘法
                auto status = cublasDgemm(handle.get(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          m, n, k,
                                          &alpha,
                                          A.data, m,
                                          B.data, k,
                                          &beta,
                                          C.data, m);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cublasDgemm failed");
                }
            }
        }
    };

    
};
#endif // DEEPX_TENSORFUNC_MATMUL_HPP