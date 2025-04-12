#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP

#include <vector>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/reduce_miaobyte.cuh"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensorfunc/authors.hpp" 
#include "deepx/tensorfunc/init_miaobyte.hpp"

namespace deepx::tensorfunc
{

    template <typename T>
    struct sumDispatcher<miaobyte, T>
    {
        static void sum(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, T(0));
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            launch_sum<T>(tensor.data, tensor.shape.strides.data(), tensor.shape.dim, tensor.shape.size,
                          reduced_dims.data(), keepdims,
                          result.data, result.shape.strides.data(), result.shape.dim);
        }
    };

    template <typename T>
    struct prodDispatcher<miaobyte, T>
    {
        static void prod(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, T(1));
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            launch_prod<T>(tensor.data, tensor.shape.strides.data(), tensor.shape.dim, tensor.shape.size,
                           reduced_dims.data(), keepdims,
                           result.data, result.shape.strides.data(), result.shape.dim);
        }
    };
    template <typename T>
    struct reducemaxDispatcher<miaobyte, T>
    {
        static void reducemax(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, std::numeric_limits<T>::lowest());
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            launch_reducemax<T>(tensor.data, tensor.shape.strides.data(), tensor.shape.dim, tensor.shape.size,
                                reduced_dims.data(), keepdims,
                                result.data, result.shape.strides.data(), result.shape.dim);
        }
    };

    template <typename T>
    struct reduceminDispatcher<miaobyte, T>
    {
        static void reducemin(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, std::numeric_limits<T>::max());
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            launch_reducemin<T>(tensor.data, tensor.shape.strides.data(), tensor.shape.dim, tensor.shape.size,
                                reduced_dims.data(), keepdims,
                                result.data, result.shape.strides.data(), result.shape.dim);
        }
    };
}

#endif //DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP