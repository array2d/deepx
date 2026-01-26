#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/cuda_math.cuh"

namespace deepx::tensorfunc
{
    // sqrt
    template <typename T>
    __global__ void sqrt_kernel(const T *A, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_sqrt(A + idx, C + idx);
        }
    }
 
    template <typename T>
    void launch_sqrt(const T *a, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        sqrt_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch sqrt kernel",cudaGetLastError());
 
    }
    template void launch_sqrt<double>(const double *a, double *c, const int size);
    template void launch_sqrt<float>(const float *a, float *c, const int size);
    template void launch_sqrt<__half>(const __half *a, __half *c, const int size);
    template void launch_sqrt<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *c, const int size);

    // pow
    template <typename T>
    __global__ void pow_kernel(const T *A, const T *B, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_pow(A + idx, B + idx, C + idx);
        }
    }
 
    template <typename T>
    void launch_pow(const T *a, const T *b, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        pow_kernel<<<numBlocks, blockSize>>>(a, b, c, size);
        throwcudaerror("Failed to launch pow kernel",cudaGetLastError());
 
    }
    template void launch_pow<double>(const double *a, const double *b, double *c, const int size);
    template void launch_pow<float>(const float *a, const float *b, float *c, const int size);

    // powscalar
    template <typename T>
    __global__ void powscalar_kernel(const T *A, const T scalar, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_pow(A + idx, &scalar, C + idx);
        }
    }
 
    template <typename T>
    void launch_powscalar(const T *a, const T scalar, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        powscalar_kernel<<<numBlocks, blockSize>>>(a, scalar, c, size);
        throwcudaerror("Failed to launch powscalar kernel",cudaGetLastError());
 
    }
    template void launch_powscalar<double>(const double *a, const double scalar, double *c, const int size);
    template void launch_powscalar<float>(const float *a, const float scalar, float *c, const int size);

    // rpowscalar
    template <typename T>
    __global__ void rpowscalar_kernel(const T scalar, const T *A, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_pow(&scalar, A + idx, C + idx);
        }
    }
 
    template <typename T>
    void launch_rpowscalar(const T scalar, const T *a, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        rpowscalar_kernel<<<numBlocks, blockSize>>>(scalar, a, c, size);
        throwcudaerror("Failed to launch rpowscalar kernel",cudaGetLastError());
    }
    template void launch_rpowscalar<double>(const double scalar, const double *a, double *c, const int size);
    template void launch_rpowscalar<float>(const float scalar, const float *a, float *c, const int size);

    // log
    template <typename T>
    __global__ void log_kernel(const T *A, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_log(A + idx, C + idx);
        }
    }
 
    template <typename T>
    void launch_log(const T *a, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        log_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch log kernel",cudaGetLastError());
    }
    template void launch_log<double>(const double *a, double *c, const int size);
    template void launch_log<float>(const float *a, float *c, const int size);
    template void launch_log<__half>(const __half *a, __half *c, const int size);
    template void launch_log<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
    // exp
    template <typename T>
    __global__ void exp_kernel(const T *A, T *C, const int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
        {
            deepx_exp(A + idx, C + idx);
        }
    }
 
    template <typename T>
    void launch_exp(const T *a, T *c, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        exp_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch exp kernel",cudaGetLastError());
    }
    template void launch_exp<double>(const double *a, double *c, const int size);
    template void launch_exp<float>(const float *a, float *c, const int size);
    template void launch_exp<__half>(const __half *a, __half *c, const int size);
    template void launch_exp<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
}
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU
