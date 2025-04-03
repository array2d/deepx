#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include <cuda/std/cmath>

namespace deepx::tensorfunc
{
    // sqrt
    template <typename T>
    __global__ void sqrt_kernel(const T *A, T *C, const int size);
    template <>
    __global__ void sqrt_kernel<double>(const double *A, double *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = sqrt(A[idx]);
        }
    }
    template <>
    __global__ void sqrt_kernel<float>(const float *A, float *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = sqrtf(A[idx]);
        }
    }

    template <>
    __global__ void sqrt_kernel<__half>(const __half *A, __half *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hsqrt(A[idx]);
        }
    }
     template <>
    __global__ void sqrt_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hsqrt(A[idx]);
        }
    }
    template <typename T>
    void launch_sqrt(int numBlocks, int blockSize, const T *a, T *c, const int size)
    {
        sqrt_kernel<<<numBlocks, blockSize>>>(a, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch sqrt kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_sqrt<double>(int numBlocks, int blockSize, const double *a, double *c, const int size);
    template void launch_sqrt<float>(int numBlocks, int blockSize, const float *a, float *c, const int size);
    template void launch_sqrt<__half>(int numBlocks, int blockSize, const __half *a, __half *c, const int size);
    template void launch_sqrt<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
    // pow
    template <typename T>
    __global__ void pow_kernel(const T *A, const T *B, T *C, const int size);
    template <>
    __global__ void pow_kernel<double>(const double *A, const double *B, double *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = pow(A[idx], B[idx]);
        }
    }
    template <>
    __global__ void pow_kernel<float>(const float *A, const float *B, float *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = powf(A[idx], B[idx]);
        }
    }

    template <typename T>
    void launch_pow(int numBlocks, int blockSize, const T *a, const T *b, T *c, const int size)
    {
        pow_kernel<<<numBlocks, blockSize>>>(a, b, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch pow kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_pow<double>(int numBlocks, int blockSize, const double *a, const double *b, double *c, const int size);
    template void launch_pow<float>(int numBlocks, int blockSize, const float *a, const float *b, float *c, const int size);

    // powscalar
    template <typename T>
    __global__ void powscalar_kernel(const T *A, const T scalar, T *C, const int size);
    template <>
    __global__ void powscalar_kernel<double>(const double *A, const double scalar, double *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = pow(A[idx], scalar);
        }
    }
    template <>
    __global__ void powscalar_kernel<float>(const float *A, const float scalar, float *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = powf(A[idx], scalar);
        }
    }
    template __global__ void powscalar_kernel<double>(const double *A, const double scalar, double *C, const int size);
    template __global__ void powscalar_kernel<float>(const float *A, const float scalar, float *C, const int size);

    template <typename T>
    void launch_powscalar(int numBlocks, int blockSize, const T *a, const T scalar, T *c, const int size)
    {
        powscalar_kernel<<<numBlocks, blockSize>>>(a, scalar, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch powscalar kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_powscalar<double>(int numBlocks, int blockSize, const double *a, const double scalar, double *c, const int size);
    template void launch_powscalar<float>(int numBlocks, int blockSize, const float *a, const float scalar, float *c, const int size);

    // log
    template <typename T>
    __global__ void log_kernel(const T *A, T *C, const int size);
    template <>
    __global__ void log_kernel<double>(const double *A, double *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = logf(A[idx]);
        }
    }
    template <>
    __global__ void log_kernel<float>(const float *A, float *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = logf(A[idx]);
        }
    }
    template <>
    __global__ void log_kernel<__half>(const __half *A, __half *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hlog(A[idx]);
        }
    }
    template <>
    __global__ void log_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {   
            C[idx] = hlog(A[idx]);
        }
    }
    
    template <typename T>
    void launch_log(int numBlocks, int blockSize, const T *a, T *c, const int size)
    {
        log_kernel<<<numBlocks, blockSize>>>(a, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch log kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_log<double>(int numBlocks, int blockSize, const double *a, double *c, const int size);
    template void launch_log<float>(int numBlocks, int blockSize, const float *a, float *c, const int size);
    template void launch_log<__half>(int numBlocks, int blockSize, const __half *a, __half *c, const int size);
    template void launch_log<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
    // exp
    template <typename T>
    __global__ void exp_kernel(const T *A, T *C, const int size);
    template <>
    __global__ void exp_kernel<double>(const double *A, double *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = exp(A[idx]);
        }
    }
    template <>
    __global__ void exp_kernel<float>(const float *A, float *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = expf(A[idx]);
        }
    }

    template <>
    __global__ void exp_kernel<__half>(const __half *A, __half *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hexp(A[idx]);
        }
    }
    template <>
    __global__ void exp_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hexp(A[idx]);
        }
    }

    template <typename T>
    void launch_exp(int numBlocks, int blockSize, const T *a, T *c, const int size)
    {
        exp_kernel<<<numBlocks, blockSize>>>(a, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch exp kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_exp<double>(int numBlocks, int blockSize, const double *a, double *c, const int size);
    template void launch_exp<float>(int numBlocks, int blockSize, const float *a, float *c, const int size);
    template void launch_exp<__half>(int numBlocks, int blockSize, const __half *a, __half *c, const int size);
    template void launch_exp<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
}
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CU
