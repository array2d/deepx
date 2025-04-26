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
#include "deepx/tensorfunc/cuda.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    struct printDispatcher<miaobyte, T>
    {
        static void print(const Tensor<T> &t, const std::string &f = "")
        {
            int64_t total_bytes = t.shape.bytes();

            // 统一分配CPU内存
            unsigned char* device_data=reinterpret_cast<unsigned char*>(t.data);
            auto [_,host_data]= device_offload(device_data,total_bytes);
            stdutil::print(t.shape.shape, host_data.get(), t.shape.dtype, f);
        };
    };

    // 特化Float16和BFloat16类型
    template <>
    struct printDispatcher<miaobyte, half>
    {
        static void print(const Tensor<half> &t, const std::string &f = "")
        {
            int64_t total_bytes = t.shape.bytes();

            // 统一分配CPU内存
            unsigned char* device_data=reinterpret_cast<unsigned char*>(t.data);
            auto [_,host_data_]= device_offload(device_data,total_bytes);
            half* host_data=reinterpret_cast<half*>(host_data_.get());
            shared_ptr<float[]> host_float(new float[t.shape.size]);
            for (size_t i = 0; i < t.shape.size; i++)
            {
                host_float[i] = __half2float(host_data[i]);
            }

            // 打印转换后的float数据
            stdutil::print(t.shape.shape, host_float.get(), Precision::Float32, f);
        }
    };

    template <>
    struct printDispatcher<miaobyte, nv_bfloat16>
    {
        static void print(const Tensor<nv_bfloat16> &t, const std::string &f = "")
        {
            int64_t total_bytes = t.shape.bytes();

            // 统一分配CPU内存
            unsigned char* device_data=reinterpret_cast<unsigned char*>(t.data);
            auto [_,host_data_]= device_offload(device_data,total_bytes);
            nv_bfloat16* host_data=reinterpret_cast<nv_bfloat16*>(host_data_.get());
            shared_ptr<float[]> host_float(new float[t.shape.size]);

            for (size_t i = 0; i < t.shape.size; i++)
            {
                host_float[i] = __bfloat162float(host_data[i]);
            }           
            // 打印转换后的float数据
            stdutil::print(t.shape.shape, host_float.get(), Precision::Float32, f); 
        }
    };

     //load
    template <typename T>
    pair<std::string,shared_ptr<Tensor<T>>> load(const std::string &path)
    {
        // 加载shape
        pair<std::string,Shape> shape_name=Shape::loadShape(path);
        Shape shape=shape_name.second;
        std::string tensor_name=shape_name.first;
 

        // 检查T 和 shape.dtype 是否匹配
        if (shape.dtype != precision<T>())
        {
            throw std::runtime_error("调用load<" + precision_str(shape.dtype) + "> 不匹配: 需要 " + precision_str(shape.dtype) +
                                     " 类型，但文件为" + precision_str(precision<T>()) + " 类型");
        }

        shared_ptr<Tensor<T>> tensor = make_shared<Tensor<T>>(New<T>(shape.shape));
        tensor->loader(path+".data",tensor->data,tensor->shape.size);
        return std::make_pair(tensor_name, tensor);
    };

}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP