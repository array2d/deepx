#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void kernel_constant(T *data, const int size, const T value)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = value;
        }
    }
    template __global__ void kernel_constant<double>(double *data, const int size, const double value);
    template __global__ void kernel_constant<float>(float *data, const int size, const float value);
    template __global__ void kernel_constant<half>(half *data, const int size, const half value);
    template __global__ void kernel_constant<nv_bfloat16>(nv_bfloat16 *data, const int size, const nv_bfloat16 value);
    template __global__ void kernel_constant<int64_t>(int64_t *data, const int size, const int64_t value);
    template __global__ void kernel_constant<int32_t>(int32_t *data, const int size, const int32_t value);
    template __global__ void kernel_constant<int16_t>(int16_t *data, const int size, const int16_t value);
    template __global__ void kernel_constant<int8_t>(int8_t *data, const int size, const int8_t value);

    template <typename T>
    void launch_constant(const int numBlocks, const int blockSize, T *a, const T value, const int size)
    {
        kernel_constant<<<numBlocks, blockSize>>>(a, size, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch constant kernel");
    }

    template void launch_constant<double>(const int numBlocks, const int blockSize,   double *a, const double value, const int size);
    template void launch_constant<float>(const int numBlocks, const int blockSize,   float *a, const float value, const int size);
    template void launch_constant<half>(const int numBlocks, const int blockSize,   half *a, const half value, const int size);
    template void launch_constant<nv_bfloat16>(const int numBlocks, const int blockSize,   nv_bfloat16 *a, const nv_bfloat16 value, const int size);
    template void launch_constant<int64_t>(const int numBlocks, const int blockSize,   int64_t *a, const int64_t value, const int size);
    template void launch_constant<int32_t>(int numBlocks, int blockSize,   int32_t *a, int32_t value, int size);
    template void launch_constant<int16_t>(const int numBlocks, const int blockSize,   int16_t *a, const int16_t value, const int size);
    template void launch_constant<int8_t>(const int numBlocks, const int blockSize,   int8_t *a, const int8_t value, const int size);

    // 添加kernel函数
    template <typename T>
    __global__ void kernel_arange(T *data, const int size, const T start, const T step)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = start + step * static_cast<T>(static_cast<float>(idx));
        }
    }
    template __global__ void kernel_arange<double>(double *data, const int size, const double start, const double step);
    template __global__ void kernel_arange<float>(float *data, const int size, const float start, const float step);
    template __global__ void kernel_arange<half>(half *data, const int size, const half start, const half step);
    template __global__ void kernel_arange<nv_bfloat16>(nv_bfloat16 *data, const int size, const nv_bfloat16 start, const nv_bfloat16 step);
    template __global__ void kernel_arange<int64_t>(int64_t *data, const int size, const int64_t start, const int64_t step);
    template __global__ void kernel_arange<int32_t>(int32_t *data, const int size, const int32_t start, const int32_t step);
    template __global__ void kernel_arange<int16_t>(int16_t *data, const int size, const int16_t start, const int16_t step);
    template __global__ void kernel_arange<int8_t>(int8_t *data, const int size, const int8_t start, const int8_t step);

    template <typename T>
    void launch_arange(const int numBlocks, const int blockSize,   T *a, const T start, const T step, const int size)
    {
        kernel_arange<<<numBlocks, blockSize>>>(a, size, start, step);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch arange kernel");
    }
    template void launch_arange<double>(const int numBlocks, const int blockSize,   double *a, const double start, const double step, const int size);
    template void launch_arange<float>(const int numBlocks, const int blockSize,   float *a, const float start, const float step, const int size);
    template void launch_arange<half>(const int numBlocks, const int blockSize,   half *a, const half start, const half step, const int size);
    template void launch_arange<nv_bfloat16>(const int numBlocks, const int blockSize,   nv_bfloat16 *a, const nv_bfloat16 start, const nv_bfloat16 step, const int size);
    template void launch_arange<int64_t>(const int numBlocks, const int blockSize,   int64_t *a, const int64_t start, const int64_t step, const int size);
    template void launch_arange<int32_t>(const int numBlocks, const int blockSize,   int32_t *a, const int32_t start, const int32_t step, const int size);
    template void launch_arange<int16_t>(const int numBlocks, const int blockSize,   int16_t *a, const int16_t start, const int16_t step, const int size);
    template void launch_arange<int8_t>(const int numBlocks, const int blockSize,   int8_t *a, const int8_t start, const int8_t step, const int size);

    // 添加kernel函数
    template <typename T>
    __global__ void kernel_uniform(T *data, const int size, const T low, const T high, const unsigned int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            // 为每个线程创建独立的随机数生成器状态
            curandState state;
            curand_init(seed, idx, 0, &state);

            // 生成[0,1)范围的随机数
            float rand = curand_uniform(&state);

            // 先用float类型进行计算，然后转换为目标类型
            float result = static_cast<float>(low) + (static_cast<float>(high) - static_cast<float>(low)) * rand;
            data[idx] = static_cast<T>(result);
        }
    }
    template __global__ void kernel_uniform<double>(double *data, const int size, const double low, const double high, const unsigned int seed);
    template __global__ void kernel_uniform<float>(float *data, const int size, const float low, const float high, const unsigned int seed);
    template __global__ void kernel_uniform<half>(half *data, const int size, const half low, const half high, const unsigned int seed);
    template __global__ void kernel_uniform<nv_bfloat16>(nv_bfloat16 *data, const int size, const nv_bfloat16 low, const nv_bfloat16 high, const unsigned int seed);
    template __global__ void kernel_uniform<int64_t>(int64_t *data, const int size, const int64_t low, const int64_t high, const unsigned int seed);
    template __global__ void kernel_uniform<int32_t>(int32_t *data, int size, int32_t low, int32_t high, unsigned int seed);
    template __global__ void kernel_uniform<int16_t>(int16_t *data, const int size, const int16_t low, const int16_t high, const unsigned int seed);
    template __global__ void kernel_uniform<int8_t>(int8_t *data, const int size, const int8_t low, const int8_t high, const unsigned int seed);

    template <typename T>
    void launch_uniform(const int numBlocks, const int blockSize, T *a, const T low, const T high, const unsigned int seed, const int size)
    {
        kernel_uniform<<<numBlocks, blockSize>>>(a, size, low, high, seed);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch uniform kernel");
    }
    template void launch_uniform<double>(const int numBlocks, const int blockSize, double *a, const double low, const double high, const unsigned int seed, const int size);
    template void launch_uniform<float>(const int numBlocks, const int blockSize, float *a, const float low, const float high, const unsigned int seed, const int size);
    template void launch_uniform<half>(const int numBlocks, const int blockSize, half *a, const half low, const half high, const unsigned int seed, const int size);
    template void launch_uniform<nv_bfloat16>(const int numBlocks, const int blockSize, nv_bfloat16 *a, const nv_bfloat16 low, const nv_bfloat16 high, const unsigned int seed, const int size);
    template void launch_uniform<int64_t>(const int numBlocks, const int blockSize, int64_t *a, const int64_t low, const int64_t high, const unsigned int seed, const int size);
    template void launch_uniform<int32_t>(const int numBlocks, const int blockSize, int32_t *a, const int32_t low, const int32_t high, const unsigned int seed, const int size);
    template void launch_uniform<int16_t>(const int numBlocks, const int blockSize, int16_t *a, const int16_t low, const int16_t high, const unsigned int seed, const int size);
    template void launch_uniform<int8_t>(const int numBlocks, const int blockSize, int8_t *a, const int8_t low, const int8_t high, const unsigned int seed, const int size);
}