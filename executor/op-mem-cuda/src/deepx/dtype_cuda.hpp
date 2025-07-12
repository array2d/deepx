#ifndef DEEPX_DTYPE_CUDA_HPP
#define DEEPX_DTYPE_CUDA_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

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


    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::BFloat16>> {
        using type = nv_bfloat16;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Float16>> {
        using type = half;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Float8E5M2>> {
        using type =  __nv_fp8_e5m2;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Float8E4M3>> {
        using type = __nv_fp8_e4m3;
    };



    template <typename T>
    struct fp8_format_map;

    template <>
    struct fp8_format_map<__nv_fp8_e5m2> {
        static constexpr __nv_fp8_interpretation_t value = __NV_E5M2;
    };

    template <>
    struct fp8_format_map<__nv_fp8_e4m3> {
        static constexpr __nv_fp8_interpretation_t value = __NV_E4M3;
    };

    template<typename T>
    struct is_fp8 : std::false_type {};                // 默认 false

    template<> struct is_fp8<__nv_fp8_e4m3> : std::true_type {};
    template<> struct is_fp8<__nv_fp8_e5m2> : std::true_type {};

    
    template <typename T>
    inline constexpr bool is_fp8_v = is_fp8<T>::value;

    template <typename T>
    struct to_half {
       static __host__ __device__ __half convert(T a) {
            return __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(a), fp8_format_map<T>::value);
       }
    };

    template <typename T>
    struct to_fp8 {
        static __host__ __device__ T convert(half a) {
            return static_cast<T>(__nv_cvt_halfraw_to_fp8(a, __NV_SATFINITE, fp8_format_map<T>::value));
        }
    };
}

#endif // DEEPX_DTYPE_CUDA_HPP
