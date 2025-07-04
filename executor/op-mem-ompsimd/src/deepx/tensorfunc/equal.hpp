#ifndef DEEPX_TENSORFUNC_EQUAL_HPP
#define DEEPX_TENSORFUNC_EQUAL_HPP
#include <cmath>
#include <omp.h>

#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    bool equal(Tensor<T> &tensor, Tensor<T> &other,float epsilon=1e-6)
    {
        bool result=true;
        if (tensor.shape.shape != other.shape.shape)
            return false;

        if constexpr (std::is_floating_point_v<T>)
        {
            #pragma omp parallel for
            for (int i = 0; i < tensor.shape.size; ++i)
            {
                if (std::fabs(tensor.data[i] - other.data[i]) > epsilon)
                {
                    #pragma omp atomic write
                    result=false;
                }
            }

            return result;
        }
        else
        {
            return std::equal(tensor.data, tensor.data + tensor.shape.size, other.data);
        }           
    };
}
#endif // DEEPX_OP_CPU_EQUAL_HPP
