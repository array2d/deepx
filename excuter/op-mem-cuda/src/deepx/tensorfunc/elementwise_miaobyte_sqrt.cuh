#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
#include <cuda_bf16.h>  
#include <cuda_fp16.h>


#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{   
    // sqrt
    template <typename T >
    __global__ void sqrt_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_sqrt(int numBlocks, int blockSize, const T* a, T* c,const int size);

    template <>
    void launch_sqrt<double>(int numBlocks, int blockSize, const double* a, double* c,const int size);

    template <>
    void launch_sqrt<float>(int numBlocks, int blockSize, const float* a, float* c,const int size);

    template <>
    void launch_sqrt<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c,const int size);

    template <>
    void launch_sqrt<__half>(int numBlocks, int blockSize, const __half* a, __half* c,const int size);

    template <>
    void launch_sqrt<int64_t>(int numBlocks, int blockSize, const int64_t* a, int64_t* c,const int size);

    template <>
    void launch_sqrt<int32_t>(int numBlocks, int blockSize, const int32_t* a, int32_t* c,const int size);

    template <>
    void launch_sqrt<int16_t>(int numBlocks, int blockSize, const int16_t* a, int16_t* c,const int size);

    template <>
    void launch_sqrt<int8_t>(int numBlocks, int blockSize, const int8_t* a, int8_t* c,const int size);

    // pow
    template <typename T>
    __global__ void pow_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_pow(int numBlocks, int blockSize, const T* a, const T* b, T* c,const int size);

    template <>
    void launch_pow<double>(int numBlocks, int blockSize, const double* a, const double* b, double* c,const int size);

    template <>
    void launch_pow<float>(int numBlocks, int blockSize, const float* a, const float* b, float* c,const int size);

    template <>
    void launch_pow<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16* b, nv_bfloat16* c,const int size);

    template <>
    void launch_pow<__half>(int numBlocks, int blockSize, const __half* a, const __half* b, __half* c,const int size);

    template <>
    void launch_pow<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t* b, int64_t* c,const int size);

    template <>
    void launch_pow<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t* b, int32_t* c,const int size);

    template <>
    void launch_pow<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t* b, int16_t* c,const int size);

    template <>
    void launch_pow<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t* b, int8_t* c,const int size);
    
    // powscalar
    template <typename T>
    __global__ void powscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_powscalar(int numBlocks, int blockSize, const T* a, const T scalar, T* c,const int size);   

    template <>
    void launch_powscalar<double>(int numBlocks, int blockSize, const double* a, const double scalar, double* c,const int size);

    template <>
    void launch_powscalar<float>(int numBlocks, int blockSize, const float* a, const float scalar, float* c,const int size);
    
    template <>
    void launch_powscalar<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16 scalar, nv_bfloat16* c,const int size);

    template <>
    void launch_powscalar<__half>(int numBlocks, int blockSize, const __half* a, const __half scalar, __half* c,const int size);

    template <>
    void launch_powscalar<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t scalar, int64_t* c,const int size);

    template <>
    void launch_powscalar<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t scalar, int32_t* c,const int size);

    template <>
    void launch_powscalar<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t scalar, int16_t* c,const int size);

    template <>
    void launch_powscalar<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t scalar, int8_t* c,const int size);

    // log
    template <typename T>
    __global__ void log_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_log(int numBlocks, int blockSize, const T* a, T* c,const int size);

    template <>
    void launch_log<double>(int numBlocks, int blockSize, const double* a, double* c,const int size);

    template <>
    void launch_log<float>(int numBlocks, int blockSize, const float* a, float* c,const int size);

    template <>
    void launch_log<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c,const int size);

        template <>
    void launch_log<__half>(int numBlocks, int blockSize, const __half* a, __half* c,const int size);

    template <>
    void launch_log<int64_t>(int numBlocks, int blockSize, const int64_t* a, int64_t* c,const int size);

    template <>
    void launch_log<int32_t>(int numBlocks, int blockSize, const int32_t* a, int32_t* c,const int size);

    template <>
    void launch_log<int16_t>(int numBlocks, int blockSize, const int16_t* a, int16_t* c,const int size);

    template <>
    void launch_log<int8_t>(int numBlocks, int blockSize, const int8_t* a, int8_t* c,const int size);

    // exp
    template <typename T>
    __global__ void exp_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_exp(int numBlocks, int blockSize, const T* a, T* c,const int size);
    
    template <>
    void launch_exp<double>(int  numBlocks, int blockSize, const double* a, double* c,const int size);

    template <>
    void launch_exp<float>(int numBlocks, int blockSize, const float* a, float* c,const int size);

    template <>
    void launch_exp<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c,const int size);

    template <>
    void launch_exp<__half>(int numBlocks, int blockSize, const __half* a, __half* c,const int size);

    template <>
    void launch_exp<int64_t>(int numBlocks, int blockSize, const int64_t* a, int64_t* c,const int size);

    template <>
    void launch_exp<int32_t>(int numBlocks, int blockSize, const int32_t* a, int32_t* c,const int size);

    template <>
    void launch_exp<int16_t>(int numBlocks, int blockSize, const int16_t* a, int16_t* c,const int size);

    template <>
    void launch_exp<int8_t>(int numBlocks, int blockSize, const int8_t* a, int8_t* c,const int size);
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
