#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP

#include <vector>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.cuh"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/shape_changeshape.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    //reshape
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

    //transpose
    template <typename T>
    struct transposeDispatcher<miaobyte, T>
    {
        static void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output)
        {
            if (dim_order.size() != tensor.shape.dim())
            {
                throw std::runtime_error("Dimension order size must match tensor dimension size for transpose");
            }
           
            launch_transpose<T>(tensor.data, tensor.shape.strides.data(),
                                output.data, output.shape.strides.data(),
                                tensor.shape.dim(), tensor.shape.size, dim_order.data());
        }
    };

    //concat        
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
                std::copy(tensors[i]->shape.strides.data(), tensors[i]->shape.strides.data() + tensors[i]->shape.dim(), std::back_inserter(inputStrides));
            }

            vector<int> shapeAtAxis(tensors.size());
            for (int i = 0; i < tensors.size(); i++)
            {
                shapeAtAxis[i] = tensors[i]->shape[axis];
            }

            launch_concat<T>(tensorsData.data(), inputStrides.data(),
                             C.data, C.shape.strides.data(),
                             C.shape.dim(),
                             C.shape.size,
                             axis, tensors.size(), shapeAtAxis.data());
        };
    };

    //broadcastTo
    template <typename T>
    struct broadcastToDispatcher<miaobyte, T>
    {
        static void broadcastTo(const Tensor<T> &A, const vector<int> &new_shape, Tensor<T> &B)
        {   
            auto A_broadcastShape = broadcastShape(A.shape.shape, new_shape);
            if (A_broadcastShape.empty()||A_broadcastShape!=new_shape)
            {
                throw TensorShapeError("Broadcast shape mismatch");
            }
            auto bmap = broadcastMap(A.shape.shape, new_shape);
            launch_broadcastTo<T>(A.data, A.shape.strides.data(), A.shape.dim(),
            bmap.data(),
            B.data, B.shape.strides.data(), B.shape.dim(), B.shape.size);
        }
    };

    //indexselectmoe_infer
    template <typename T,typename GatherAxisT>
    struct indexselectDispatcher<miaobyte, T,GatherAxisT>
    {
        static void indexselect(const Tensor<T> &input, const Tensor<GatherAxisT> &indices, const int axis, Tensor<T> &output){
            int gatherAxis = axis < 0 ? input.shape.dim() + axis : axis;
            vector<int> gatherShape = indexselectShape(input.shape.shape, indices.shape.shape, gatherAxis);
            if (gatherShape.empty()||gatherShape!=output.shape.shape)
            {
                throw TensorShapeError("Indexselect shape mismatch");
            }
            
            launch_indexselect<T,GatherAxisT>(input.data, input.shape.strides.data(), input.shape.dim(),
                            indices.data, indices.shape.strides.data(), indices.shape.dim(),
                            gatherAxis,
                            output.data,output.shape.strides.data(),output.shape.dim(),output.shape.size);
        }
    };

    //repeat
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
            launch_repeat<T>(A.data, A.shape.strides.data(), 
                             repeats.data(), 
                             B.data, B.shape.strides.data(),B.shape.size, B.shape.dim());
        }
    };
}
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP