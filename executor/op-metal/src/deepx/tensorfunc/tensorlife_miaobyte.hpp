#ifndef DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP

#include <algorithm>
#include <string>

#include "stdutil/fs.hpp"
#include "deepx/tensor.hpp"
#include "deepx/dtype_mps.hpp"
#include "deepx/tensorfunc/tensorlife.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    static T *newFn(int size)
    {
        return new T[size];
    }

    template <typename T>
    static void freeFn(T *data)
    {
        delete[] data;
    }

    template <typename T>
    static void copyFn(T *src, T *dest, int size)
    {
        std::copy(src, src + size, dest);
    }

    template <typename T>
    static void saveFn(T *data, size_t size, const std::string &path)
    {
        auto *udata = reinterpret_cast<unsigned char *>(data);
        size_t bytes = size * sizeof(T);
        stdutil::save(udata, bytes, path);
    }

    template <typename T>
    static void loadFn(const std::string &path, T *data, int size)
    {
        auto *udata = reinterpret_cast<unsigned char *>(data);
        size_t bytes = size * sizeof(T);
        stdutil::load(path, udata, bytes);
    }

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
    }

    template <typename T>
    void copy(const Tensor<T> &src, Tensor<T> &dst)
    {
        dst.shape = src.shape;
        dst.copyer(src.data, dst.data, src.shape.size);
    }
}

#endif // DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
