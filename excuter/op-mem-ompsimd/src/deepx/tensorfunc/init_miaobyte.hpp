#ifndef DEEPX_TENSORFUNC_INIT_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAOBYTE_HPP

#include <cmath>
#include <random>
#include <omp.h>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/init.hpp"

namespace deepx::tensorfunc
{   
    //constant
    template <typename T>
    struct constantDispatcher<miaobyte, T>
    {
        static void constant(Tensor<T> &tensor, const T value)
        {
            std::fill(tensor.data, tensor.data + tensor.shape.size, value);
        }
    };

    //uniform
    template <typename T>
    struct uniformDispatcher<miaobyte, T>
    {
        static void uniform(Tensor<T> &tensor, const T low = 0, const T high = 1, const unsigned int seed = 0)
        {
            std::uniform_real_distribution<double> distribution(low, high);
            std::default_random_engine generator;
            
            // 设置随机数生成器种子
            if (seed == 0)
            {
                std::random_device rd;
                generator.seed(rd());
            }
            else
            {
                generator.seed(seed);
            }

            // 单线程循环填充数据
            for (int i = 0; i < tensor.shape.size; ++i)
            {
                tensor.data[i] = static_cast<T>(distribution(generator));
            }
        }
    };

    //arange
    template <typename T>
    struct arangeDispatcher<miaobyte, T>
    {
        static void arange(Tensor<T> &tensor, const T start, const T step = 1)
        {
            // 单线程循环填充数据
            for (int i = 0; i < tensor.shape.size; ++i)
            {
                tensor.data[i] = start + i * step;
            }
        }
    };

    //normal
    template <typename T>
    struct normalDispatcher<miaobyte, T>
    {
        static void normal(Tensor<T> &tensor, const T mean, const T stddev, const unsigned int seed = 0)
        {
            std::normal_distribution<double> dist(mean, stddev);
            std::default_random_engine generator;
            
            // 设置随机数生成器种子
            if (seed == 0)
            {
                std::random_device rd;
                generator.seed(rd());
            }
            else
            {
                generator.seed(seed);
            }

            // 单线程循环填充数据
            for (int i = 0; i < tensor.shape.size; ++i)
            {   
                tensor.data[i] = static_cast<T>(dist(generator));   
            }
        }
    };  
}

#endif // DEEPX_OP_CPU_INIT_HPP