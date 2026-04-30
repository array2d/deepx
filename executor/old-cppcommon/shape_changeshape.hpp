#ifndef DEEPX_SHAPE_CHANGESHAPE_HPP
#define DEEPX_SHAPE_CHANGESHAPE_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"
#include "stdutil/error.hpp"

namespace deepx
{
    // transpose

    using namespace std;
    std::vector<int> swaplastTwoDimOrder(const std::vector<int> &shape);

    std::vector<int> transposeShape(const std::vector<int> &shape, const std::vector<int> &dimOrder);

    // concat
    Shape concatShape(const std::vector<Shape> &shapes, const int axis);

    template <typename T>
    Shape concatShape(const std::vector<Tensor<T> *> &tensors, const int axis)
    {
        std::vector<Shape> shapes;
        for (int i = 0; i < tensors.size(); ++i)
        {
            shapes.push_back(tensors[i]->shape);
        }
        return concatShape(shapes, axis);
    }

    template <typename T>
    bool checkShapeConcat(const std::vector<Tensor<T> *> &tensors, const int axis, const Tensor<T> &output)
    {
        int axisDim = 0;
        for (int i = 0; i < tensors.size(); i++)
        {
            if (tensors[i]->shape.dim() != output.shape.dim())
            {
                throw TensorShapeError("All input tensors must have the same dimension size for concat");
            }
            for (int j = 0; j < tensors[i]->shape.dim(); j++)
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

    // broadcast
    std::vector<int> broadcastShape(const std::vector<int> &a, const std::vector<int> &b);
    enum BroadcastMap
    {
        xTox = 0,
        nullTo1 = 1,
        xTo1 = 2,
    };
    std::vector<BroadcastMap> broadcastMap(const std::vector<int> &a, const std::vector<int> &b);

 
    //indexselect
    vector<int> indexselectShape(const vector<int> &input_shape, const vector<int> &index_shape, const int axis);

    //repeat
    std::vector<int> repeatShape(const std::vector<int> &src, const std::vector<int> &repeats);
}
#endif // DEEPX_SHAPE_CHANGESHAPE_HPP