#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>

#include "shape_reduce.hpp"
#include "tensor.hpp"
#include "tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // ═══════════════════════════════════════════════════════════
    // Helper: compute output index from input indices
    // ═══════════════════════════════════════════════════════════
    static int computeOutputIndex(const std::vector<int> &input_indices,
                                  const std::vector<int> &reduced_dims,
                                  bool keepdims,
                                  const Shape &output_shape)
    {
        std::vector<int> out_indices;
        for (size_t i = 0; i < input_indices.size(); ++i)
        {
            if (reduced_dims[i] == 0)
            {
                out_indices.push_back(input_indices[i]);
            }
            else if (keepdims && (reduced_dims[i] == 1))
            {
                out_indices.push_back(0);
            }
        }
        return output_shape.linearat(out_indices);
    }

    // ═══════════════════════════════════════════════════════════
    // sum
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct sumDispatcher<miaobyte, T>
    {
        static void sum(const Tensor<T> &tensor, const std::vector<int> &dims,
                        const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, T(0));

            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);

            for (int64_t i = 0; i < tensor.shape.size; ++i)
            {
                std::vector<int> indices = tensor.shape.linearto(static_cast<int>(i));
                int outputIdx = computeOutputIndex(indices, reduced_dims, keepdims, result.shape);
                result.data[outputIdx] += tensor.data[i];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // prod
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct prodDispatcher<miaobyte, T>
    {
        static void prod(const Tensor<T> &tensor, const std::vector<int> &dims,
                         const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, T(1));

            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);

            for (int64_t i = 0; i < tensor.shape.size; ++i)
            {
                std::vector<int> indices = tensor.shape.linearto(static_cast<int>(i));
                int outputIdx = computeOutputIndex(indices, reduced_dims, keepdims, result.shape);
                result.data[outputIdx] *= tensor.data[i];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // reducemax
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct reducemaxDispatcher<miaobyte, T>
    {
        static void reducemax(const Tensor<T> &tensor, const std::vector<int> &dims,
                              const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, std::numeric_limits<T>::lowest());

            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);

            for (int64_t i = 0; i < tensor.shape.size; ++i)
            {
                std::vector<int> indices = tensor.shape.linearto(static_cast<int>(i));
                int outputIdx = computeOutputIndex(indices, reduced_dims, keepdims, result.shape);
                result.data[outputIdx] = std::max(result.data[outputIdx], tensor.data[i]);
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // reducemin
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct reduceminDispatcher<miaobyte, T>
    {
        static void reducemin(const Tensor<T> &tensor, const std::vector<int> &dims,
                              const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, std::numeric_limits<T>::max());

            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);

            for (int64_t i = 0; i < tensor.shape.size; ++i)
            {
                std::vector<int> indices = tensor.shape.linearto(static_cast<int>(i));
                int outputIdx = computeOutputIndex(indices, reduced_dims, keepdims, result.shape);
                result.data[outputIdx] = std::min(result.data[outputIdx], tensor.data[i]);
            }
        }
    };

} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
