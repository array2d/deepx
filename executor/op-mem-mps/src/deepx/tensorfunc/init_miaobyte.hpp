#ifndef DEEPX_TENSORFUNC_INIT_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAOBYTE_HPP

#include <algorithm>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // fill/constant
    template <typename T>
    struct constantDispatcher<miaobyte, T>
    {
        static void constant(Tensor<T> &tensor, const T value)
        {
            std::fill(tensor.data, tensor.data + tensor.shape.size, value);
        }
    };
}

#endif // DEEPX_TENSORFUNC_INIT_MIAOBYTE_HPP
