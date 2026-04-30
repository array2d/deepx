#ifndef DEEPX_SHAPE_SUM_HPP
#define DEEPX_SHAPE_SUM_HPP

#include "deepx/shape.hpp"

namespace deepx
{

        // 检查dims参数是否合法,返回整理后的dims
        std::vector<int> checkedDims(const std::vector<int> &inputshape, const std::vector<int> &dims);

        // 返回求和后的形状     
        std::vector<int> reducedShape(const std::vector<int> &inputshape, const std::vector<int> &dims, const bool keepdim = false);

        // 返回需要求和的维度
        std::vector<int> reducedDim(const std::vector<int> &inputshape, const std::vector<int> &dims );
}

#endif // DEEPX_SHAPE_SUM_HPP