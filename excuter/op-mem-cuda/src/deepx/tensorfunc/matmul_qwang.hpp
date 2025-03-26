#ifndef DEEPX_TENSORFUNC_MATMUL_WQING_HPP
#define DEEPX_TENSORFUNC_MATMUL_WQING_HPP

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/matmul_qwang.cuh"

namespace deepx::tensorfunc
{
    using namespace deepx;

    template <typename T>
    struct matmulDispatcher<qwang, T>
    {
        static void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            launch_matmul(C.data, A.data, B.data, A.shape[-2], A.shape[-1], B.shape[-1]);
        }
    };

 
}
#endif  