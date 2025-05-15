#ifndef DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP

#include <cuda_runtime.h>
#include <stdexcept>

#include "stdutil/fs.hpp"
#include "deepx/tensor.hpp"
#include "deepx/dtype_cuda.hpp"
#include "deepx/tensorfunc/tensorlife.hpp"
#include "deepx/tensorfunc/cuda.hpp"
// 具体的张量类
namespace deepx::tensorfunc
{
    // NewFn
    template <typename T>
    static T *newFn(int size)
    {
        T *data;
        cudaError_t err = cudaMalloc(&data, size * sizeof(T));
        if (err != cudaSuccess)
        {   
            throwcudaerror("Failed to cudaMalloc "+std::to_string(size) +" "+ precision_str(precision<T>()),err);
        }
        return data;
    }

    template <typename T>
    static void freeFn(T *data)
    {
        cudaFree(data);
    }

    template <typename T>
    static void copyFn(T *src, T *dest, int size)
    {
        cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    template <typename T>
    static void saveFn(T *tensorData, size_t size, const std::string &path)
    {
        // 保存data
        int64_t total_bytes = size * sizeof(T);

        // 统一分配CPU内存

        auto [_,host_data] = device_offload(reinterpret_cast<unsigned char*>(tensorData), total_bytes);
        stdutil::save(host_data.get(), total_bytes, path);
    }

    // 不做任何转换，直接从内存到文件，或从文件到内存
    template <typename T>
    static void loadFn(const std::string &path, T *data, int count)
    {
        auto [file_size, hostdata] = stdutil::load(path);
        if (file_size != count * sizeof(T))
        {
            Precision p = precision<T>();
            throw std::runtime_error("file_size!=count*" + precision_str(p));
        }
        cudaMemcpy(data, hostdata.get(), file_size, cudaMemcpyHostToDevice);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to copy data from host to device");
        }
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

    // rename

}
#endif // DEEPX_TENSORFUNC_TENSORLIFE_MIAOBYTE_HPP
