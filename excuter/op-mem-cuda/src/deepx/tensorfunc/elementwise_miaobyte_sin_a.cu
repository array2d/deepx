#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_A_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_A_CU

#include <cuda_bf16.h>

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // sin
    template <typename T>
    __global__ void sin_kernel(const T *A, T *C, const int size);

    template <>
    __global__ void sin_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hsin(A[idx]);
        }
    }

    template <typename T>
    void launch_sin(int numBlocks, int blockSize, const T *a, T *c, const int size)
    {
        sin_kernel<<<numBlocks, blockSize>>>(a, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch sin kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    template void launch_sin<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);

    // cos
    template <typename T>
    __global__ void cos_kernel(const T *A, T *C, const int size);

    template <>
    __global__ void cos_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hcos(A[idx]);
        }
    }
    template <typename T>
    void launch_cos(int numBlocks, int blockSize, const T *a, T *c, const int size)
    {
        cos_kernel<<<numBlocks, blockSize>>>(a, c, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch cos kernel: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    template void launch_cos<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16 *a, nv_bfloat16 *c, const int size);

}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_A_CU
