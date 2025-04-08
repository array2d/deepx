#ifndef DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP

#include <iostream>
#include <string>
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
            if (host_data == nullptr) {
                throw std::runtime_error("Failed to allocate host memory");
            }
            
            // 统一复制数据到CPU
            cudaError_t err = cudaMemcpy(host_data, t.data, total_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                delete[] host_data;
                throw std::runtime_error("Failed to copy data from device to host");
            }

            // 对于half和bf16类型需要转换为float
            if (t.shape.dtype == Precision::Float16 || t.shape.dtype == Precision::BFloat16) {
                float* host_float = new float[t.shape.size];
                if (host_float == nullptr) {
                    delete[] host_data;
                    throw std::runtime_error("Failed to allocate host memory for float conversion");
                }

                // 在CPU上进行类型转换
                if (t.shape.dtype == Precision::Float16) {
                    for(size_t i = 0; i < t.shape.size; i++) {
                        host_float[i] = __half2float(((half*)host_data)[i]);
                    }
                } else { // BFloat16
                    for(size_t i = 0; i < t.shape.size; i++) {
                        host_float[i] = __bfloat162float(((nv_bfloat16*)host_data)[i]);
                    }
                }

                // 打印转换后的float数据
                stdutil::print(t.shape.shape, host_float, Precision::Float32, f.empty() ? "%.4f" : f);
                delete[] host_float;
            }
            else {
                // 其他类型直接打印
                stdutil::print(t.shape.shape, host_data, t.shape.dtype, f);
            }

            delete[] host_data;
        }
    };
}

#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP