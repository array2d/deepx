#include <vector>
#include <stdexcept>

#include "deepx/shape_changeshape.hpp"

namespace deepx 
{
    //transpose

        std::vector<int> swaplastTwoDimOrder(const std::vector<int> &shape) 
    {
        vector<int> dimOrder = shape;
        std::iota(dimOrder.begin(), dimOrder.end(), 0);
        swap(dimOrder[dimOrder.size() - 1], dimOrder[dimOrder.size() - 2]);
        return dimOrder;
    }
    std::vector<int> transposeShape(const std::vector<int> &shape, const std::vector<int> &dimOrder) 
    {
        if (dimOrder.size() != shape.size())
        {
            throw std::invalid_argument("dimOrder size does not match the number of dimensions in the TensorCPU.");
        }
        std::vector<int> newShape = shape;
        for (size_t i = 0; i < dimOrder.size(); ++i)
        {
            newShape[i] =shape[dimOrder[i]];
        }
        return newShape;
    }

    //concat

    Shape concatShape(const std::vector<Shape> &shapes,const int axis){
        std::vector<int> outputShape(shapes[0].dim);
        outputShape=shapes[0].shape;
        for (int i = 1; i < shapes.size(); ++i)
        {
            if (shapes[i].dim != outputShape.size())
            {
                throw std::invalid_argument("All tensors must have the same number of dimensions.");
            }
            for (size_t j = 0; j < outputShape.size(); ++j)
            {
                if (j == axis)
                {
                    outputShape[j] += shapes[i].shape[j];
                }
                else if (shapes[i].shape[j] != outputShape[j])
                {
                    throw std::invalid_argument("Shapes of tensors must match except in the concatenation axis.");
                }
            }
        }
        return Shape(outputShape);
    }

   //broadcast
    std::vector<int> broadcastShape(const std::vector<int> &a, const std::vector<int> &b)
    {
        int len1 = a.size();
        int len2 = b.size();
        int maxLen = std::max(len1, len2);
        std::vector<int> result(maxLen);

        for (int i = 1; i <= maxLen; ++i)
        {
            int dim1 = (i <= len1) ? a[len1 - i] : 1;
            int dim2 = (i <= len2) ? b[len2 - i] : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                result.clear();
                return result;
            }
            result[maxLen - i] = std::max(dim1, dim2);
        }
        return result;
    }
    std::vector<BroadcastMap> broadcastMap(const std::vector<int> &shape, const std::vector<int> &broadcastShape)
    {
        std::vector<BroadcastMap> broadcastMap(broadcastShape.size());
        int s = broadcastShape.size() - shape.size();
        for (int i = 0; i < s; ++i)
        {
            broadcastMap[i] = nullTo1;
        }
        for (int i = s; i < broadcastShape.size(); ++i)
        {
            if (shape[i - s] == broadcastShape[i])
            {
                broadcastMap[i] = xTox;
            }
            else if (shape[i - s] == 1)
            {
                broadcastMap[i] = xTo1;
            }
            else
            {
                throw std::runtime_error("Shapes are not broadcastable for operation");
            }
        }
        return broadcastMap;
    }

    void fromBroadcastIndices(const std::vector<BroadcastMap> &broadcastMap, const std::vector<int> &broadcastIndices, std::vector<int> &oldIndices )
    {
        for (int i = 0, j = 0; i < broadcastIndices.size(); ++i)
        {
            switch (broadcastMap[i])
            {
            case xTox:
                oldIndices[j++] = broadcastIndices[i];
                break;
            case nullTo1:
                break;
            case xTo1:
                oldIndices[j++] = 0;
                break;
            }
        }
    }
}