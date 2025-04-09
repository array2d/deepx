#ifndef DEEPX_TENSORFUNC_HIGHWAY_HPP
#define DEEPX_TENSORFUNC_HIGHWAY_HPP

#include <hwy/highway.h>

namespace deepx::tensorfunc
{
    using namespace hwy::HWY_NAMESPACE;

    template <typename T, class D>
    T ReduceMul(D d, Vec<D> v)
    {
        T result = GetLane(v);
        for (size_t i = 1; i < Lanes(d); ++i)
        {
            result *= ExtractLane(v, i);
        }
        return result;
    }

}

#endif
