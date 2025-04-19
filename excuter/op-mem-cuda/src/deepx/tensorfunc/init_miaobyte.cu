#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/cuda.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void kernel_constant(T *data, const T value, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            data[idx] = value;
        }
    }

    template <typename T>
    void launch_constant(T *a, const T value, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        kernel_constant<<<numBlocks, blockSize>>>(a, value, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch constant kernel");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to synchronize device");
    }

    template void launch_constant<double>(double *a, const double value, const int size);
    template void launch_constant<float>(float *a, const float value, const int size);
    template void launch_constant<half>(half *a, const half value, const int size);
    template void launch_constant<nv_bfloat16>(nv_bfloat16 *a, const nv_bfloat16 value, const int size);
    template void launch_constant<int64_t>(int64_t *a, const int64_t value, const int size);
    template void launch_constant<int32_t>(int32_t *a, const int32_t value, const int size);
    template void launch_constant<int16_t>(int16_t *a, const int16_t value, const int size);
    template void launch_constant<int8_t>(int8_t *a, const int8_t value, const int size);
    template void launch_constant<bool>(bool *a, const bool value, const int size);

    // 添加kernel函数
    template <typename T>
    __global__ void kernel_arange(T *data, const float start, const float step, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            data[idx] = static_cast<T>(start + step * static_cast<float>(idx));
        }
    }

    template <typename T>
    void launch_arange(T *a, const T start, const T step, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        kernel_arange<<<numBlocks, blockSize>>>(a, static_cast<float>(start), static_cast<float>(step), size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch arange kernel");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to synchronize device");
    }
    template void launch_arange<double>(double *a, const double start, const double step, const int size);
    template void launch_arange<float>(float *a, const float start, const float step, const int size);
    template void launch_arange<half>(half *a, const half start, const half step, const int size);
    template void launch_arange<nv_bfloat16>(nv_bfloat16 *a, const nv_bfloat16 start, const nv_bfloat16 step, const int size);
    template void launch_arange<int64_t>(int64_t *a, const int64_t start, const int64_t step, const int size);
    template void launch_arange<int32_t>(int32_t *a, const int32_t start, const int32_t step, const int size);
    template void launch_arange<int16_t>(int16_t *a, const int16_t start, const int16_t step, const int size);
    template void launch_arange<int8_t>(int8_t *a, const int8_t start, const int8_t step, const int size);

    // 添加kernel函数
    template <typename T>
    __global__ void kernel_uniform(T *data, const float low, const float high, const unsigned int seed, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        curandState state;
        curand_init(seed, threadIdx.x, 0, &state); // 仅初始化一次

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            // 生成[0,1)范围的随机数
            float rand = curand_uniform(&state);

            // 先用float类型进行计算，然后转换为目标类型
            float result = low + (high - low) * rand;
            data[idx] = static_cast<T>(result);
        }
    }

    template <typename T>
    void launch_uniform(T *a, const T low, const T high, const unsigned int seed, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        kernel_uniform<<<numBlocks, blockSize>>>(a, float(low), float(high), seed, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch uniform kernel");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to synchronize device");
    }
    template void launch_uniform<double>(double *a, const double low, const double high, const unsigned int seed, const int size);
    template void launch_uniform<float>(float *a, const float low, const float high, const unsigned int seed, const int size);
    template void launch_uniform<half>(half *a, const half low, const half high, const unsigned int seed, const int size);
    template void launch_uniform<nv_bfloat16>(nv_bfloat16 *a, const nv_bfloat16 low, const nv_bfloat16 high, const unsigned int seed, const int size);
    template void launch_uniform<int64_t>(int64_t *a, const int64_t low, const int64_t high, const unsigned int seed, const int size);
    template void launch_uniform<int32_t>(int32_t *a, const int32_t low, const int32_t high, const unsigned int seed, const int size);
    template void launch_uniform<int16_t>(int16_t *a, const int16_t low, const int16_t high, const unsigned int seed, const int size);
    template void launch_uniform<int8_t>(int8_t *a, const int8_t low, const int8_t high, const unsigned int seed, const int size);

    // normal
    template <typename T>
    __global__ void kernel_normal(T *data, const float mean, const float stddev, const unsigned int seed, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        curandState state;
        curand_init(seed, threadIdx.x, 0, &state); // 仅初始化一次

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            // 生成[0,1)范围的随机数
            float rand = curand_normal(&state);
            // 先用float类型进行计算，然后转换为目标类型
            float result = mean + stddev * rand;
            data[idx] = static_cast<T>(result);
        }
    }
    template <typename T>
    void launch_normal(T *a, const T mean, const T stddev, const unsigned int seed, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        kernel_normal<<<numBlocks, blockSize>>>(a,float(mean), float(stddev), seed, size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch normal kernel");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to synchronize device");
    }
    template void launch_normal<double>(double *a, const double mean, const double stddev, const unsigned int seed, const int size);
    template void launch_normal<float>(float *a, const float mean, const float stddev, const unsigned int seed, const int size);
    template void launch_normal<half>(half *a, const half mean, const half stddev, const unsigned int seed, const int size);
    template void launch_normal<nv_bfloat16>(nv_bfloat16 *a, const nv_bfloat16 mean, const nv_bfloat16 stddev, const unsigned int seed, const int size);
    template void launch_normal<int64_t>(int64_t *a, const int64_t mean, const int64_t stddev, const unsigned int seed, const int size);
    template void launch_normal<int32_t>(int32_t *a, const int32_t mean, const int32_t stddev, const unsigned int seed, const int size);
    template void launch_normal<int16_t>(int16_t *a, const int16_t mean, const int16_t stddev, const unsigned int seed, const int size);
    template void launch_normal<int8_t>(int8_t *a, const int8_t mean, const int8_t stddev, const unsigned int seed, const int size);

}