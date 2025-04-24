#ifndef DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP

#include "stdutil/fs.hpp"
#include "deepx/tensorfunc/tensorlife.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensor.hpp"
#include "deepx/dtype.hpp"
#include "deepx/dtype_ompsimd.hpp"
#include "deepx/tensorfunc/new_mempool.hpp"

namespace deepx::tensorfunc
{

    template <typename T>
    static T *newFn(int size)
    {
        return static_cast<T *>(MemoryPool::Malloc(size * sizeof(T)));
    }

    template <typename T>
    static void freeFn(T *data)
    {
        MemoryPool::Free(data);
    }

    template <typename T>
    static void copyFn(T *data, T *data2, int size)
    {
        std::copy(data, data + size, data2);
    }

    template <typename T>
    static void saveFn(T *data, size_t size, const std::string &path)
    {   
        unsigned char *udata = reinterpret_cast<unsigned char *>(data);
        size_t udatasize = size * sizeof(T);
        stdutil::save(udata,udatasize,path);
    }
    

    template <typename T>
    static void loadFn(const std::string &path, T *data, int size)
    {
        unsigned char *udata = reinterpret_cast<unsigned char *>(data);
        size_t udatasize = size * sizeof(T);
        stdutil::load(path,udata,udatasize);
    }
    // New
    template <typename T>
    Tensor<T> New(const std::vector<int> &shapedata)
    {
        Shape shape(shapedata);
        shape.dtype = precision<T>();

        Tensor<T> tensor(shape);
        tensor.deleter = freeFn<T>;
        tensor.copyer = copyFn<T>;
        tensor.newer = newFn<T>;
        tensor.saver = saveFn<T>;
        tensor.loader = loadFn<T>;


        tensor.data = newFn<T>(shape.size);
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