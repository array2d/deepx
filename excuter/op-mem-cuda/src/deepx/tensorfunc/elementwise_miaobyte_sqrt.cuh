#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
#include <cuda_bf16.h>  
#include <cuda_fp16.h>

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

    
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
