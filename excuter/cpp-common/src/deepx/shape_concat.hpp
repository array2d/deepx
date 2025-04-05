#ifndef DEEPX_SHAPE_CONCAT_HPP
#define DEEPX_SHAPE_CONCAT_HPP

#include "deepx/shape.hpp"
#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx
{

    Shape concatShape(const std::vector<Shape> &shapes,const int axis);

    template<typename T>
    Shape concatShape(const std::vector<Tensor<T>*> &tensors,const int axis){
        std::vector<Shape> shapes;
        for (int i = 0; i < tensors.size(); ++i)
        {
            shapes.push_back(tensors[i]->shape);
        }
        return concatShape(shapes,axis);
    }

    template<typename T>
    bool checkShapeConcat(const std::vector<Tensor<T>*> &tensors,const int axis,const Tensor<T> &output){
        int axisDim=0;
        for (int i = 0; i < tensors.size(); i++)
        {
            if (tensors[i]->shape.dim != output.shape.dim)
            {
                throw TensorShapeError("All input tensors must have the same dimension size for concat");
            }
            for (int j = 0; j < tensors[i]->shape.dim; j++)
            {
                if (j != axis)
                {   
                    if (tensors[i]->shape[j] != output.shape[j])
                    {
                        throw TensorShapeError("All input tensors must have the same dimension size for concat");
                    }
                }
                else
                {
                    axisDim += tensors[i]->shape[j];
                }
            }
        }
        return axisDim == output.shape[axis];
    }
};
#endif // DEEPX_SHAPE_CONCAT_HPP