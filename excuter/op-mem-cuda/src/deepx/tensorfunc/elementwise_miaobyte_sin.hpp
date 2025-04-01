#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SIN_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SIN_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_sin.cuh"

#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
 
    template <typename T>
    struct sinDispatcher<miaobyte, T>
    {
        static void sin(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("sin");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_sin(numBlocks, blockSize, A.data, C.data, A.shape.size);           
        }   
    };

    template <typename T>
    struct cosDispatcher<miaobyte, T>
    {
        static void cos(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("cos");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_cos(numBlocks, blockSize, A.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct tanDispatcher<miaobyte, T>
    {
        static void tan(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("tan");
            }
            const int blockSize = A.shape.size > 256 ? 256 : A.shape.size;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_tan(numBlocks, blockSize, A.data, C.data, A.shape.size);
        }
    };

   
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
