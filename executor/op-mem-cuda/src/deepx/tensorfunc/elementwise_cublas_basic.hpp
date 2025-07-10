#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_CUBLAS_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_CUBLAS_BASIC_HPP

#include <cuda_fp16.h> // 为了支持half精度
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/cuda.hpp"
namespace deepx::tensorfunc
{

    // double特化
    template <>
    struct addDispatcher<cublas, double>
    {
        static void add(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size)
            {
                throw std::runtime_error("Tensor shapes must match for addition");
            }

            static CublasHandle handle;
            const double alpha = 1.0;
            const double beta = 1.0;
            auto status = cublasDgeam(handle.get(),
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      A.shape.size, 1,
                                      &alpha,
                                      A.data, A.shape.size,
                                      &beta,
                                      B.data, B.shape.size,
                                      C.data, C.shape.size);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cuBLAS Dgeam failed");
            }
        }
    };
    // float特化
    template <>
    struct addDispatcher<cublas, float>
    {
        static void add(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size)
            {
                throw std::runtime_error("Tensor shapes must match for addition");
            }

            static CublasHandle handle;
            const float alpha = 1.0f;
            const float beta = 1.0f;

            // 使用cublasSgeam直接计算 C = alpha*A + beta*B
            auto status = cublasSgeam(handle.get(),
                                      CUBLAS_OP_N,     // 不转置A
                                      CUBLAS_OP_N,     // 不转置B
                                      A.shape.size, 1, // 假设是向量（或展平处理）
                                      &alpha,
                                      A.data, A.shape.size,
                                      &beta,
                                      B.data, B.shape.size,
                                      C.data, C.shape.size);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cuBLAS Sgeam failed");
            }
        }
    };

}

#endif