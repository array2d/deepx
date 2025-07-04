#ifndef DEEPX_DTYPE_CUDA_HPP
#define DEEPX_DTYPE_CUDA_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "deepx/dtype.hpp"

namespace deepx
{
    using namespace std;
        // 获取类型对应的Precision
    template <typename T>
    constexpr Precision precision()
    {
        if constexpr (std::is_same_v<T, double>)
            return Precision::Float64;
        else if constexpr (std::is_same_v<T, float>)
            return Precision::Float32;
        else if constexpr (std::is_same_v<T, half>) return Precision::Float16;
        else if constexpr (std::is_same_v<T, nv_bfloat16>) return Precision::BFloat16;
        else if constexpr (std::is_same_v<T, int64_t>)
            return Precision::Int64;
        else if constexpr (std::is_same_v<T, int32_t>)
            return Precision::Int32;
        else if constexpr (std::is_same_v<T, int16_t>)
            return Precision::Int16;
        else if constexpr (std::is_same_v<T, int8_t>)
            return Precision::Int8;
        else if constexpr (std::is_same_v<T, bool>)
            return Precision::Bool;
        else if constexpr (std::is_same_v<T, std::string>)
            return Precision::String;
        else
            return Precision::Any;
    }   
}

#endif // DEEPX_DTYPE_CUDA_HPP
