#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CUH

 #include <cuda_bf16.h>  
#include <cuda_fp16.h>


#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // sin
    template <typename T>
    __global__ void sin_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_sin(const T* a, T* c, const int size);

    
    template <typename T>
    __global__ void cos_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_cos( const T* a, T* c, const int size);

    // tan
    template <typename T>
    __global__ void tan_kernel(const T* A, T* C, const int size);

    template <typename T>
    void launch_tan( const T* a, T* c, const int size);
}

#endif