#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_A_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_A_CU

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
    template void launch_sqrt<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);

    // log
    template <typename T>
    __global__ void log_kernel(const T *A, T *C, const int size);

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
    template void launch_log<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);

    // exp
    template <typename T>
    __global__ void exp_kernel(const T *A, T *C, const int size);

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
    template void launch_exp<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_A_CU
