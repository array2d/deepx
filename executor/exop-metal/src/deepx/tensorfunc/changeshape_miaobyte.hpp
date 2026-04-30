#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP

#include <stdexcept>
#include <vector>
#include <cstring>

#include "deepx/tensor.hpp"
#include "deepx/shape_changeshape.hpp"
#include "tensorfunc/changeshape.hpp"
#include "tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // ═══════════════════════════════════════════════════════════
    // reshape — pure CPU, checks element count & copies data
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct reshapeDispatcher<miaobyte, T>
    {
        static void reshape(const Tensor<T> &tensor, const std::vector<int> &shape, Tensor<T> &output)
        {
            int new_prod = 1;
            for (int dim : shape)
            {
                new_prod *= dim;
            }

            if (tensor.shape.size != new_prod)
            {
                throw std::invalid_argument("Shape size mismatch");
            }
            Shape newshape(shape);
            output.shape.shape  = newshape.shape;
            output.shape.strides = newshape.strides;
            output.shape.size    = newshape.size;

            if (tensor.data != output.data)
            {
                std::memcpy(output.data, tensor.data, tensor.shape.size * sizeof(T));
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // transpose
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct transposeDispatcher<miaobyte, T>
    {
        static void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output)
        {
            if (dim_order.size() != static_cast<size_t>(tensor.shape.dim()))
            {
                throw std::invalid_argument("dimOrder size does not match the number of dimensions.");
            }
            if (output.shape.size != tensor.shape.size)
            {
                throw std::runtime_error("transpose error: output shape size mismatch");
            }

            std::vector<int> new_shape = transposeShape(tensor.shape.shape, dim_order);
            output.shape = Shape(new_shape);

            int ndim = tensor.shape.dim();
            std::vector<int> src_indices(ndim, 0);
            for (int64_t i = 0; i < output.shape.size; ++i)
            {
                std::vector<int> dst_indices = output.shape.linearto(static_cast<int>(i));
                for (size_t j = 0; j < static_cast<size_t>(ndim); ++j)
                {
                    src_indices[dim_order[j]] = dst_indices[j];
                }
                int src_linear = tensor.shape.linearat(src_indices);
                output.data[i] = tensor.data[src_linear];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // concat
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct concatDispatcher<miaobyte, T>
    {
        static void concat(const std::vector<Tensor<T>*> tensors, const int axis, Tensor<T> &result)
        {
            if (!checkShapeConcat(tensors, axis, result))
            {
                throw TensorShapeError("Output tensor shape must match sum of input shapes for concat");
            }

            int dimC = axis + 1;
            int copylen = tensors[0]->shape.strides[axis];

            for (int64_t idx = 0; idx < result.shape.size; idx += copylen)
            {
                std::vector<int> indices = result.shape.linearto(static_cast<int>(idx));

                int concatIdx = indices[axis];
                int tensorIdx = 0;
                while (tensorIdx < static_cast<int>(tensors.size()))
                {
                    if (concatIdx < tensors[tensorIdx]->shape[axis])
                    {
                        break;
                    }
                    concatIdx -= tensors[tensorIdx]->shape[axis];
                    tensorIdx++;
                }

                std::vector<int> src_indices = indices;
                src_indices[axis] = concatIdx;
                int src_idx = tensors[tensorIdx]->shape.linearat(src_indices);
                std::memcpy(result.data + idx, tensors[tensorIdx]->data + src_idx, copylen * sizeof(T));
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // broadcastTo helper
    // ═══════════════════════════════════════════════════════════
    static std::vector<int> fromBroadcastIndices(const std::vector<BroadcastMap> &bmap,
                                                  const std::vector<int> &broadcastIndices)
    {
        std::vector<int> srcindices;
        for (size_t i = 0; i < bmap.size(); ++i)
        {
            switch (bmap[i])
            {
            case xTox:
                srcindices.push_back(broadcastIndices[i]);
                break;
            case nullTo1:
                break;
            case xTo1:
                srcindices.push_back(0);
                break;
            }
        }
        return srcindices;
    }

    // ═══════════════════════════════════════════════════════════
    // broadcastTo
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct broadcastToDispatcher<miaobyte, T>
    {
        static void broadcastTo(const Tensor<T> &A, const std::vector<int> &new_shape, Tensor<T> &B)
        {
            auto A_broadcastShape = broadcastShape(A.shape.shape, new_shape);
            if (A_broadcastShape.empty() || A_broadcastShape != new_shape)
            {
                throw TensorShapeError("Broadcast shape mismatch");
            }
            auto bmap = broadcastMap(A.shape.shape, new_shape);

            int ndim = static_cast<int>(new_shape.size());
            for (int64_t i = 0; i < B.shape.size; ++i)
            {
                std::vector<int> bindices = B.shape.linearto(static_cast<int>(i));
                std::vector<int> aindices = fromBroadcastIndices(bmap, bindices);
                B.data[i] = A.data[A.shape.linearat(aindices)];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // indexselect helper
    // ═══════════════════════════════════════════════════════════
    template <typename GatherAxisT>
    static void fromIndexselectIndices(const std::vector<int> &output_indices,
                                        const Tensor<GatherAxisT> &index,
                                        std::vector<int> &index_indices,
                                        const int gatherAxis,
                                        std::vector<int> &input_indices)
    {
        std::copy(output_indices.begin(), output_indices.begin() + gatherAxis, input_indices.begin());
        std::copy(output_indices.begin() + gatherAxis,
                  output_indices.begin() + gatherAxis + static_cast<int>(index_indices.size()),
                  index_indices.begin());
        int index_idx = index.shape.linearat(index_indices);
        input_indices[gatherAxis] = index.data[index_idx];
        std::copy(output_indices.begin() + gatherAxis + static_cast<int>(index_indices.size()),
                  output_indices.begin() + static_cast<int>(output_indices.size()),
                  input_indices.begin() + gatherAxis + 1);
    }

    // ═══════════════════════════════════════════════════════════
    // indexselect
    // ═══════════════════════════════════════════════════════════
    template <typename T, typename GatherAxisT>
    struct indexselectDispatcher<miaobyte, T, GatherAxisT>
    {
        static void indexselect(const Tensor<T> &input, const Tensor<GatherAxisT> &index,
                                const int axis, Tensor<T> &output)
        {
            int gatherAxis = axis < 0 ? input.shape.dim() + axis : axis;
            if (gatherAxis < 0 || gatherAxis >= input.shape.dim())
            {
                throw std::invalid_argument("Axis is out of bounds");
            }

            std::vector<int> gatherShape = indexselectShape(input.shape.shape, index.shape.shape, gatherAxis);
            if (gatherShape.empty() || gatherShape != output.shape.shape)
            {
                throw TensorShapeError("Indexselect shape mismatch");
            }

            std::vector<int> input_indices(input.shape.dim(), 0);
            std::vector<int> index_indices(index.shape.dim(), 0);

            for (int64_t i = 0; i < output.shape.size; ++i)
            {
                std::vector<int> output_indices = output.shape.linearto(static_cast<int>(i));
                fromIndexselectIndices(output_indices, index, index_indices, gatherAxis, input_indices);
                output.data[i] = input.data[input.shape.linearat(input_indices)];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════
    // repeat
    // ═══════════════════════════════════════════════════════════
    template <typename T>
    struct repeatDispatcher<miaobyte, T>
    {
        static void repeat(const Tensor<T> &A, const std::vector<int> &repeats, Tensor<T> &B)
        {
            auto new_shape = repeatShape(A.shape.shape, repeats);
            if (new_shape.empty() || new_shape != B.shape.shape)
            {
                throw TensorShapeError("Repeat shape mismatch");
            }

            int ndim = A.shape.dim();
            std::vector<int> src_indices(ndim, 0);
            for (int64_t i = 0; i < B.shape.size; ++i)
            {
                std::vector<int> indices = B.shape.linearto(static_cast<int>(i));
                for (size_t d = 0; d < static_cast<size_t>(ndim); ++d)
                {
                    src_indices[d] = indices[d] / repeats[d];
                }
                B.data[i] = A.data[A.shape.linearat(src_indices)];
            }
        }
    };

} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP
