#ifndef DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP

#include <iostream>
#include <string>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <deepx/tensor.hpp>
#include <deepx/dtype.hpp>
#include <stdutil/vector.hpp>
#include <stdutil/print.hpp>
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/io.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    struct printDispatcher<miaobyte, T>
    {
        static void print(const Tensor<T> &t, const std::string &f = "")
        {
            int bytes = precision_bits(t.shape.dtype) / 8;
            size_t total_bytes = t.shape.size * bytes;

            // 统一分配CPU内存
            unsigned char *host_data = new unsigned char[total_bytes];
            if (host_data == nullptr)
            {
                throw std::runtime_error("Failed to allocate host memory");
            }

            stdutil::print(t.shape.shape, host_data, t.shape.dtype, f);
            delete[] host_data;
        };
    };

    // 特化Float16和BFloat16类型
    template <>
    struct printDispatcher<miaobyte, half>
    {
        static void print(const Tensor<half> &t, const std::string &f = "")
        {
            int bytes = precision_bits(t.shape.dtype) / 8;
            size_t total_bytes = t.shape.size * bytes;

            // 统一分配CPU内存
            unsigned char *host_data = new unsigned char[total_bytes];
            if (host_data == nullptr)
            {
                throw std::runtime_error("Failed to allocate host memory");
            }

            // 统一复制数据到CPU
            cudaError_t err = cudaMemcpy(host_data, t.data, total_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                delete[] host_data;
                throw std::runtime_error("Failed to copy data from device to host");
            }

            float *host_float = new float[t.shape.size];
            if (host_float == nullptr)
            {
                delete[] host_data;
                throw std::runtime_error("Failed to allocate host memory for float conversion");
            }

            for (size_t i = 0; i < t.shape.size; i++)
            {
                host_float[i] = __half2float(((half *)host_data)[i]);
            }

            delete[] host_data;
            // 打印转换后的float数据
            stdutil::print(t.shape.shape, host_float, Precision::Float32, f);
            delete[] host_float;
        }
    };

    template <>
    struct printDispatcher<miaobyte, nv_bfloat16>
    {
        static void print(const Tensor<nv_bfloat16> &t, const std::string &f = "")
        {
            int bytes = precision_bits(t.shape.dtype) / 8;
            size_t total_bytes = t.shape.size * bytes;

            // 统一分配CPU内存
            unsigned char *host_data = new unsigned char[total_bytes];
            if (host_data == nullptr)
            {
                throw std::runtime_error("Failed to allocate host memory");
            }

            // 统一复制数据到CPU
            cudaError_t err = cudaMemcpy(host_data, t.data, total_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                delete[] host_data;
                throw std::runtime_error("Failed to copy data from device to host");
            }

            float *host_float = new float[t.shape.size];
            if (host_float == nullptr)
            {
                delete[] host_data;
                throw std::runtime_error("Failed to allocate host memory for float conversion");
            }

            for (size_t i = 0; i < t.shape.size; i++)
            {
                host_float[i] = __bfloat162float(((nv_bfloat16 *)host_data)[i]);
            }
            delete[] host_data;
            // 打印转换后的float数据
            stdutil::print(t.shape.shape, host_float, Precision::Float32, f);
            delete[] host_float;
        }
    };

    template <typename T>
    void save(Tensor<T> &tensor, const std::string &path)
    {
        // 保存shape
        std::string shapepath = path + ".shape";
        std::string shapedata = tensor.shape.toYaml();
        std::ofstream shape_fs(shapepath, std::ios::binary);
        shape_fs.write(shapedata.c_str(), shapedata.size());
        shape_fs.close();

        // 保存data
        int bytes = precision_bits(tensor.shape.dtype) / 8;
        size_t total_bytes = tensor.shape.size * bytes;

        // 统一分配CPU内存
        unsigned char *host_data = new unsigned char[total_bytes];
        if (host_data == nullptr)
        {
            throw std::runtime_error("Failed to allocate host memory");
        }

        // 统一复制数据到CPU
        cudaError_t err = cudaMemcpy(host_data, tensor.data, total_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            delete[] host_data;
            throw std::runtime_error("Failed to copy data from device to host");
        }

        std::string datapath = path + ".data";
        std::ofstream data_fs(datapath, std::ios::binary | std::ios::in | std::ios::out);

        if (!data_fs.is_open())
        {
            // 如果文件不存在，则创建新文件
            data_fs.open(datapath, std::ios::binary | std::ios::out);
        }
        data_fs.seekp(0);
        data_fs.write(reinterpret_cast<const char *>(host_data), total_bytes);
        data_fs.close();

        delete[] host_data;
    };

    template <typename T>
    pair<std::string, shared_ptr<Tensor<T>>> load(const std::string &path)
    {
        // 加载shape
        pair<std::string, Shape> shape_name = loadShape(path);
        Shape shape = shape_name.second;
        std::string tensor_name = shape_name.first;

        // 检查T 和 shape.dtype 是否匹配
        if (shape.dtype != precision<T>())
        {
            throw std::runtime_error("调用load<" + precision_str(shape.dtype) + "> 不匹配: 需要 " + precision_str(shape.dtype) +
                                     " 类型，但文件为" + precision_str(precision<T>()) + " 类型");
        }

        // 检查file.size，是否是tensor.size*sizeof(T)
        std::string datapath = path + ".data";
        std::ifstream data_fs(datapath, std::ios::binary);
        data_fs.seekg(0, std::ios::end);
        std::streamsize fileSize = data_fs.tellg();
        std::streamsize expectedSize = shape.size * precision_bits(shape.dtype) / 8;

        if (fileSize != expectedSize)
        {
            throw std::runtime_error("数据文件大小不足: 需要 " + std::to_string(expectedSize) +
                                     " 字节，但文件只有 " + std::to_string(fileSize) + " 字节");
        }
        data_fs.seekg(0);

        // TODO 从文件，到cuda内存（可能是显存）

        shared_ptr<Tensor<T>> tensor = make_shared<Tensor<T>>(New<T>(shape.shape));
        unsigned char *host_data = new unsigned char[fileSize];
        if (host_data == nullptr)
        {
            throw std::runtime_error("Failed to allocate host memory");
        }
        data_fs.read(reinterpret_cast<char *>(host_data), fileSize);
        data_fs.close();

        cudaError_t err = cudaMemcpy(tensor->data, host_data, fileSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            delete[] host_data;
            throw std::runtime_error("Failed to copy data from host to device");
        }
        delete[] host_data;
        return std::make_pair(tensor_name, tensor);
    }
}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP