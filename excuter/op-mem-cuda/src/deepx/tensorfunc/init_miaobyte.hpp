#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP

#include <random>

#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/init_miaobyte.cuh"
namespace deepx::tensorfunc
{
    // constant
    template <typename T>
    struct constantDispatcher<miaobyte, T>
    {
        static void constant(Tensor<T> &tensor, const T value)
        {
            launch_constant(tensor.data, value, tensor.shape.size);
        }
    };

    // arange
    template <typename T>
    struct arangeDispatcher<miaobyte, T>
    {
        static void arange(Tensor<T> &tensor, const T start, const T step)
        {
            launch_arange(tensor.data, start, step, tensor.shape.size);
        }
    };

    // uniform
    template <typename T>
    struct uniformDispatcher<miaobyte, T>
    {
        static void uniform(Tensor<T> &tensor, const T low, const T high, const unsigned int seed)
        {
            launch_uniform(tensor.data, low, high, seed, tensor.shape.size);
        }
    };

    // normal
    template <typename T>
    struct normalDispatcher<miaobyte, T>
    {
        static void normal(Tensor<T> &tensor, const T mean, const T stddev, const unsigned int seed)
        {
            launch_normal(tensor.data, mean, stddev, seed, tensor.shape.size);
        }
    };
}

#endif
