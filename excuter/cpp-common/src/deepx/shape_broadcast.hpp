#ifndef DEEPX_OP_CPU_SHAPE_HPP
#define DEEPX_OP_CPU_SHAPE_HPP


#include "deepx/tensor.hpp"


namespace deepx  {
    
    std::vector<int> broadcastShape(const std::vector<int> &a, const std::vector<int> &b);
    enum BroadcastMap
    {
        xTox = 0,
        nullTo1 = 1,
        xTo1 = 2,
    };
    std::vector<BroadcastMap> broadcastMap(const std::vector<int> &a, const std::vector<int> &b);

}

#endif // DEEPX_OP_CPU_SHAPE_HPP