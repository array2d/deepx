#ifndef DEEPX_OP_CUDA_NEW_HPP
#define DEEPX_OP_CUDA_NEW_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/dtype_cuda.hpp"
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
    Tensor<T> New(const std::vector<int> &shapedata, T *data = nullptr)
    {
        Shape shape(shapedata);
        shape.dtype=precision<T>();
        Tensor<T> tensor(shape);
        tensor.device = CUDA; // 使用 CUDA 设备
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;

        if (data != nullptr) {
            tensor.data = data;
        } else {
            tensor.data = dataNew<T>(shape.size);
        }
        return tensor;
    }

    template <typename T>
    Tensor<T> New(const std::initializer_list<int> &shapedata, T *data = nullptr)
    {
        std::vector<int> shape_vec(shapedata);
        Shape shape(shape_vec);
        shape.dtype=precision<T>();
        Tensor<T> tensor(shape);
        tensor.device = CUDA; // 使用 CUDA 设备
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;

        if (data != nullptr) {
            tensor.data = data;
        } else {
            tensor.data = dataNew<T>(shape.size);
        }
        return tensor;
    }

    template <typename T>
    Tensor<T> clone(const Tensor<T> &tensor)
    {
        Tensor<T> result = New<T>(tensor.shape.shape);
        tensor.copyer(tensor.data, result.data, tensor.shape.size);
        return result;
    }
}
#endif // DEEPX_OP_CUDA_NEW_HPP
