#ifndef DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/dtype_cuda.hpp"
#include "deepx/tensorfunc/tensorlife.hpp"
// 具体的张量类
namespace deepx::tensorfunc
{
    template <typename T>
    static T* dataNew(int size)
    {
        T* data;
        cudaError_t err = cudaMalloc(&data, size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate Unified Memory");
        }
        return data;
    }

    template <typename T>
    static void dataFree(T *data)
    {
        cudaFree(data);
    }

    template <typename T>
    static void dataCopy(T *data, T *data2, int size)
    {
        cudaMemcpy(data2, data, size * sizeof(T), cudaMemcpyDefault);
    }

    template <typename T>
    Tensor<T> New(const std::vector<int> &shapedata)
    {
        Shape shape(shapedata);
        shape.dtype=precision<T>();
        Tensor<T> tensor(shape);
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;

        tensor.data = dataNew<T>(shape.size);
        return tensor;
    }
 
    template <typename T>
    void copy(const Tensor<T> &src,Tensor<T> &dst)
    {
        dst.shape=src.shape;
        dst.copyer(src.data, dst.data, src.shape.size);
    }

    //rename

}
#endif // DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
