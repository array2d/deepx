#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP

#include <vector>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.cuh"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/shape_concat.hpp"
namespace deepx::tensorfunc
{
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
            if (tensor.data == output.data)
            {
                output.shape.shape=newshape.shape;
                output.shape.strides=newshape.strides;
            }
            else
            {
                output.shape.shape=newshape.shape;
                output.shape.strides=newshape.strides;
                output.copyer(tensor.data, output.data, tensor.shape.size);
            }
        }
    };

    template <typename T>
    struct transposeDispatcher<miaobyte, T>
    {
        static void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output)
        {
            if (dim_order.size() != tensor.shape.dim)
            {
                throw std::runtime_error("Dimension order size must match tensor dimension size for transpose");
            }
            auto [actual_blocks, optimal_block_size] = BestDims(tensor.shape.size);
            launch_transpose<T>(actual_blocks, optimal_block_size,
                                tensor.data, tensor.shape.strides.data(),
                                output.data, output.shape.strides.data(),
                                tensor.shape.dim, tensor.shape.size, dim_order.data());
        }
    };

    template <typename T>
    struct concatDispatcher<miaobyte, T>
    {
        static void concat(const vector<Tensor<T> *> tensors, const int axis, Tensor<T> &C)
        {
            // checkshape
            if (!checkShapeConcat(tensors, axis, C))
            {
                throw TensorShapeError("Output tensor shape size must match the sum of input tensor shape sizes for concat");
            }

            vector<const T *> tensorsData(tensors.size());
            for (int i = 0; i < tensors.size(); i++)
            {
                tensorsData[i] = tensors[i]->data;
            }

            vector<int> inputStrides;
            for (int i = 0; i < tensors.size(); i++)
            {
                std::copy(tensors[i]->shape.strides.data(), tensors[i]->shape.strides.data() + tensors[i]->shape.dim, std::back_inserter(inputStrides));
            }

            vector<int> shapeAtAxis(tensors.size());
            for (int i = 0; i < tensors.size(); i++)
            {
                shapeAtAxis[i] = tensors[i]->shape[axis];
            }

            launch_concat<T>(tensorsData.data(), inputStrides.data(),
                             C.data, C.shape.strides.data(),
                             C.shape.dim,
                             C.shape.size,
                             axis, tensors.size(), shapeAtAxis.data());
        };
    };
}
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP