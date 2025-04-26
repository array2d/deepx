#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "stdutil/error.hpp"
#include "deepx/shape_reduce.hpp"

namespace deepx
{
    std::vector<int> checkedDims(const std::vector<int> &inputshape, const std::vector<int> &dims)
    {
        std::vector<int> checkeddims;
        // 如果dims为空，则求和所有维度
        if (dims.empty())
        {
            for (int i = 0; i < inputshape.size(); ++i)
            {
                checkeddims.push_back(i);
            }
        }
        else
        {   
            // 验证维度
            for (int d : dims)
            {
                if (d < 0)
                {
                    d = inputshape.size() + d;
                }
                if (d >= inputshape.size())
                {
                    throw TensorShapeError("Dimension out of range in sum");
                }
                checkeddims.push_back(d);
            }
        }

        // 排序
        std::sort(checkeddims.begin(), checkeddims.end());
        // 去重
        checkeddims.erase(std::unique(checkeddims.begin(), checkeddims.end()), checkeddims.end());

        return checkeddims;
    }

    std::vector<int> reducedShape(const std::vector<int> &inputshape, const std::vector<int> &dims, const bool keepdim)
    {

        // 创建一个映射数组，标记哪些维度需要求和
        std::vector<int> reducedims = reducedDim(inputshape, dims);

        // 计算输出形状
        std::vector<int> outputShape;

        for (size_t i = 0; i < inputshape.size(); ++i)
        {
            if (reducedims[i] == 0)
            {
                outputShape.push_back(inputshape[i]);
            }
            else if (keepdim)
            {
                outputShape.push_back(1);
            }
        }

        // 如果所有维度都被求和，返回标量张量
        if (outputShape.empty())
        {
            outputShape.push_back(1);
        }
        return outputShape;
    }

    // 创建一个(map映射)数组，标记哪些维度需要求和
    std::vector<int> reducedDim(const std::vector<int> &shape, const std::vector<int> &dims)
    {
        std::vector<int> reducdMap(shape.size(), 0);
        for (int dim : dims)
        {
            reducdMap[dim] = 1;
        }
        return reducdMap;
    }
}