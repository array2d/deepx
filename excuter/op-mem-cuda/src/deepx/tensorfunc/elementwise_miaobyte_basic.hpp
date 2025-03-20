#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_basic.cuh"

#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    // CUDA kernel函数声明
   

    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) {
                throw TensorShapeError("add");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_add(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);
           
        }   
    };

    template <typename T>
    struct addscalarDispatcher<miaobyte, T>
    {
        static void addscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("addscalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_addscalar(numBlocks, blockSize, A.data, scalar, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct subDispatcher<miaobyte, T>
    {
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) { 
                throw TensorShapeError("sub");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_sub(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct subscalarDispatcher<miaobyte, T>
    {
        static void subscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("subscalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_subscalar(numBlocks, blockSize, A.data, scalar, C.data, A.shape.size);
        }
    };  
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
