#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_compare.cuh"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    // CUDA kernel函数声明
   

    template <typename T>
    struct maxDispatcher<miaobyte, T>
    {
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("max");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_max(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);           
        }   
    };

    template <typename T>
    struct maxscalarDispatcher<miaobyte, T>
    {
        static void maxscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("maxscalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_maxscalar(numBlocks, blockSize, A.data, scalar, C.data, A.shape.size);
        }
    };
    
    template <typename T>
    struct minDispatcher<miaobyte, T>
    {
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("min");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_min(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct minscalarDispatcher<miaobyte, T>
    {
        static void minscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("minscalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_minscalar(numBlocks, blockSize, A.data, scalar, C.data, A.shape.size);
        }
    };
    template <typename T>
    struct compareDispatcher<miaobyte, T>
    {
        static void compare(const Tensor<T> &A, const Tensor<T> &B, Tensor<float> &mask)
        {
            if (A.shape.size != B.shape.size || A.shape.size != mask.shape.size) { 
                throw TensorShapeError("compare");  
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_compare(numBlocks, blockSize, A.data, B.data, mask.data, A.shape.size);
        }
    };

    template <typename T>
    struct comparescalarDispatcher<miaobyte, T>
    {
        static void comparescalar(const Tensor<T> &A, const T scalar, Tensor<float> &mask)
        {
            if (A.shape.size != mask.shape.size) {
                throw TensorShapeError("comparescalar");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_comparescalar(numBlocks, blockSize, A.data, scalar, mask.data, A.shape.size);
        }
    };
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
