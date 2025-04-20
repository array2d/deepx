#ifndef DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP

#include "deepx/tensorfunc/tensorlife.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensor.hpp"
#include "deepx/dtype.hpp"
#include "deepx/dtype_ompsimd.hpp"
#include "deepx/tensorfunc/new_mempool.hpp"

namespace deepx::tensorfunc
{

    template <typename T>
    static T *dataNew(int size)
    {
        return static_cast<T *>(MemoryPool::Malloc(size * sizeof(T)));
    }

    template <typename T>
    static void dataFree(T *data)
    {
        MemoryPool::Free(data);
    }

    template <typename T>
    static void dataCopy(T *data, T *data2, int size)
    {
        std::copy(data, data + size, data2);
    }

    // New
    template <typename T>
    Tensor<T> New(const std::vector<int> &shapedata)
    {
        Shape shape(shapedata);
        shape.dtype = precision<T>();

        Tensor<T> tensor(shape);
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;
        tensor.data = dataNew<T>(shape.size);
        return tensor;
    };

    template <typename T>
    void copy(const Tensor<T> &src, Tensor<T> &dst)
    {
        dst.shape = src.shape;
        dst.copyer(src.data, dst.data, src.shape.size);
    }

}
#endif // DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP