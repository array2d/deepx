#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP

#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size ||
                A.shape.shape != B.shape.shape || A.shape.shape != C.shape.shape)
            {
                throw std::invalid_argument("shape mismatch");
            }
            for (int64_t i = 0; i < A.shape.size; ++i)
            {
                C.data[i] = A.data[i] + B.data[i];
            }
        }
    };
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
