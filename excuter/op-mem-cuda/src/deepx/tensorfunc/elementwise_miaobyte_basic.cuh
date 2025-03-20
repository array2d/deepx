#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH

#include <cuda_bf16.h>  
#include <cuda_fp16.h>


#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
     template <typename T>
    __global__ void add_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_add(int numBlocks, int blockSize, const T* a, const T* b, T* c,const int size);

    template <>
    void launch_add<double>(int numBlocks, int blockSize, const double* a, const double* b, double* c,const int size);

    template <>
    void launch_add<float>(int numBlocks, int blockSize, const float* a, const float* b, float* c,const int size);

    template <>
    void launch_add<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16* b, nv_bfloat16* c,const int size);

    template <>
    void launch_add<__half>(int numBlocks, int blockSize, const __half* a, const __half* b, __half* c,const int size);

    template <>
    void launch_add<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t* b, int64_t* c,const int size);

    template <>
    void launch_add<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t* b, int32_t* c,const int size);

    template <>
    void launch_add<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t* b, int16_t* c,const int size);

    template <>
    void launch_add<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t* b, int8_t* c,const int size);  


    // addscalar
     template <typename T>
    __global__ void addscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_addscalar(int numBlocks, int blockSize, const T* a, const T scalar, T* c,const int size);

    template <>
    void launch_addscalar<double>(int numBlocks, int blockSize, const double* a, const double scalar, double* c,const int size);

    template <>
    void launch_addscalar<float>(int numBlocks, int blockSize, const float* a, const float scalar, float* c,const int size);

    template <>
    void launch_addscalar<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16 scalar, nv_bfloat16* c,const int size);

    template <>
    void launch_addscalar<__half>(int numBlocks, int blockSize, const __half* a, const __half scalar, __half* c,const int size);

    template <>
    void launch_addscalar<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t scalar, int64_t* c,const int size);

    template <>
    void launch_addscalar<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t scalar, int32_t* c,const int size);

    template <>
    void launch_addscalar<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t scalar, int16_t* c,const int size);

    template <>
    void launch_addscalar<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t scalar, int8_t* c,const int size);

    // sub
    template <typename T>
    __global__ void sub_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_sub(int numBlocks, int blockSize, const T* a, const T* b, T* c,const int size);  

    template <>
    void launch_sub<double>(int numBlocks, int blockSize, const double* a, const double* b, double* c,const int size);

    template <> 
    void launch_sub<float>(int numBlocks, int blockSize, const float* a, const float* b, float* c,const int size);

    template <>
    void launch_sub<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16* b, nv_bfloat16* c,const int size);

    template <> 
    void launch_sub<__half>(int numBlocks, int blockSize, const __half* a, const __half* b, __half* c,const int size);

    template <>
    void launch_sub<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t* b, int64_t* c,const int size);

    template <> 
    void launch_sub<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t* b, int32_t* c,const int size);

    template <> 
    void launch_sub<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t* b, int16_t* c,const int size);

    template <> 
    void launch_sub<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t* b, int8_t* c,const int size);

    // subscalar
    template <typename T>
    __global__ void subscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_subscalar(const int numBlocks, const int blockSize, const T* a, const T scalar, T* c,const int size);

    template <>
    void launch_subscalar<double>(const int numBlocks, const int blockSize, const double* a, const double scalar, double* c,const int size);

    template <>
    void launch_subscalar<float>(const int numBlocks, const int blockSize, const float* a, const float scalar, float* c,const int size);

    template <>
    void launch_subscalar<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16* a, const nv_bfloat16 scalar, nv_bfloat16* c,const int size);

    template <>
    void launch_subscalar<__half>(const int numBlocks, const int blockSize, const __half* a, const __half scalar, __half* c,const int size);

    template <>
    void launch_subscalar<int64_t>(const int numBlocks, const int blockSize, const int64_t* a, const int64_t scalar, int64_t* c,const int size);

    template <>
    void launch_subscalar<int32_t>(const int numBlocks, const int blockSize, const int32_t* a, const int32_t scalar, int32_t* c,const int size);

    template <>
    void launch_subscalar<int16_t>(const int numBlocks, const int blockSize, const int16_t* a, const int16_t scalar, int16_t* c,const int size);

    template <>
    void launch_subscalar<int8_t>(const int numBlocks, const int blockSize, const int8_t* a, const int8_t scalar, int8_t* c,const int size);    
 
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
