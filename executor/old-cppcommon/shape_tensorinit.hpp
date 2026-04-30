#ifndef DEEPX_SHAPE_TENSORINIT_HPP
#define DEEPX_SHAPE_TENSORINIT_HPP

#include "deepx/shape.hpp"

namespace deepx
{
    std::pair<int, int> calculateFanInAndFanOut(const Shape &shape);
}

#endif // DEEPX_SHAPE_TENSORINIT_HPP