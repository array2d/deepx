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
            launch_sin(A.data, C.data, A.shape.size);           
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
            launch_cos(A.data, C.data, A.shape.size);
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
            launch_tan(A.data, C.data, A.shape.size);
        }
    };

   
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
