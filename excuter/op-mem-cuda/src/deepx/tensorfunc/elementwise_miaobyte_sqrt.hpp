#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_sqrt.cuh"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    // CUDA kernel函数声明
   

    template <typename T>
    struct sqrtDispatcher<miaobyte, T>
    {
        static void sqrt(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("sqrt");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_sqrt(numBlocks, blockSize, A.data, C.data, A.shape.size);           
        }   
    };

    template <typename T>
    struct powDispatcher<miaobyte, T>
    {
        static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("pow");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_pow(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct powscalarDispatcher<miaobyte, T>
    {
        static void powscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("powscalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_powscalar(numBlocks, blockSize, A.data, scalar, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct logDispatcher<miaobyte, T>
    {
        static void log(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("log");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_log(numBlocks, blockSize, A.data, C.data, A.shape.size);
        }
    };  

    template <typename T>
    struct expDispatcher<miaobyte, T>
    {
        static void exp(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("exp");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_exp(numBlocks, blockSize, A.data, C.data, A.shape.size);
        }
    };

    
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
