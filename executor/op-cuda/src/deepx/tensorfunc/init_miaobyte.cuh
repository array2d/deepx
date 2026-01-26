#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/init.hpp"

namespace deepx::tensorfunc
{
    //填充
    template <typename T>
    __global__ void kernel_constant(T *data, const T value, const int size);

    template <typename T>
    void launch_constant(T *a, const T value, const int size);
    
    //dropout
    template <typename T>
    __global__ void dropout_kernel(T* A, const float p,const unsigned int seed,const int size);

    template <typename T>
    void launch_dropout(T* a, const float p,const unsigned int seed,const int size);

    //初始化
    //arange
    template <typename T>
    __global__ void kernel_arange(T *data, const float start, const float step, const int size);

    template <typename T>
    void launch_arange(T *a, const T start, const T step, const int size);

    //uniform
    template <typename T>
    __global__ void kernel_uniform(T *data, const float low, const float high, const unsigned int seed, const int size);

    template <typename T>
    void launch_uniform(T *a, const T low, const T high, const unsigned int seed, const int size);

    //normal
    template <typename T>
    __global__ void kernel_normal(T *data, const float mean, const float stddev, const unsigned int seed, const int size);

    template <typename T>
    void launch_normal(T *a, const T mean, const T stddev, const unsigned int seed, const int size);

}

#endif