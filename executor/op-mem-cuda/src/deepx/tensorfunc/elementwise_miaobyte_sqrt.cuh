#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{   
    // sqrt
    template <typename T >
    __global__ void sqrt_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_sqrt(const T* a, T* c,const int size);
 
    
    // pow
    template <typename T>
    __global__ void pow_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_pow(const T* a, const T* b, T* c,const int size);
 
     
    // powscalar
    template <typename T>
    __global__ void powscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_powscalar(const T* a, const T scalar, T* c,const int size);   

    // rpowscalar
    template <typename T>
    __global__ void rpowscalar_kernel(const T scalar, const T* A, T* C, const int size);

    template <typename T>
    void launch_rpowscalar(const T scalar, const T* a, T* c, const int size);
    
    // log
    template <typename T>
    __global__ void log_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_log(const T* a, T* c,const int size);
 
    // exp
    template <typename T>
    __global__ void exp_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_exp(const T* a, T* c,const int size);
    
   
    
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_SQRT_CUH
