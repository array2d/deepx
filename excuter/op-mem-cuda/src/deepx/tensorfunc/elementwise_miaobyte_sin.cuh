#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CUH

 #include <cuda_bf16.h>  
#include <cuda_fp16.h>


#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void sin_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_sin(int numBlocks, int blockSize, const T* a, T* c, const int size);

    template <>
    void launch_sin<double>(int numBlocks, int blockSize, const double* a, double* c, const int size);

    template <>
    void launch_sin<float>(int numBlocks, int blockSize, const float* a, float* c, const int size);

    template <>
    void launch_sin<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c, const int size);

    template <>
    void launch_sin<__half>(int numBlocks, int blockSize, const __half* a, __half* c, const int size);
    
    template <typename T>
    __global__ void cos_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_cos(int numBlocks, int blockSize, const T* a, T* c, const int size);

    template <>
    void launch_cos<double>(int numBlocks, int blockSize, const double* a, double* c, const int size);

    template <>
    void launch_cos<float>(int numBlocks, int blockSize, const float* a, float* c, const int size);

    template <>
    void launch_cos<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c, const int size);

    template <>
    void launch_cos<__half>(int numBlocks, int blockSize, const __half* a, __half* c, const int size);
    
    template <typename T>
    __global__ void tan_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_tan(int numBlocks, int blockSize, const T* a, T* c, const int size);

    template <>
    void launch_tan<double>(int numBlocks, int blockSize, const double* a, double* c, const int size);

    template <>
    void launch_tan<float>(int numBlocks, int blockSize, const float* a, float* c, const int size);

    template <>
    void launch_tan<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, nv_bfloat16* c, const int size);

    template <>
    void launch_tan<__half>(int numBlocks, int blockSize, const __half* a, __half* c, const int size);
 
}

#endif