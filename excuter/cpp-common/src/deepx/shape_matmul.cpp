#include <stdexcept>

#include "deepx/shape_matmul.hpp"

namespace deepx
{
    Shape matmul_shape(const Shape &A, const Shape &B)
    {
        if (A.dim() < 2 || B.dim() < 2)
        {
            throw std::invalid_argument("A and B must >= 2D tensors");
        }
        if (A[-1] != B[-2])
        {
            throw std::invalid_argument("A[-1] must be equal to B[-2]");
        }
        std::vector<int> resultshape(A.dim());
        std::copy(A.shape.begin(), A.shape.begin() + A.dim(), resultshape.begin());
        Shape result(resultshape);
        result[-1] = B[-1];
        return result;
    }
}