#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH


#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{

    //todtype
    template <typename T,typename Dtype>
    __global__ void todtype_kernel(const T* A, Dtype* C,const int size);

    template <typename T,typename Dtype>
    void launch_todtype(const T* a, Dtype* c,const int size);

    //add
     template <typename T>
    __global__ void add_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_add(const T* a, const T* b, T* c,const int size);

 

    // addscalar
     template <typename T>
    __global__ void addscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_addscalar(const T* a, const T scalar, T* c,const int size);
 
    // sub
    template <typename T>
    __global__ void sub_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_sub(const T* a, const T* b, T* c,const int size);  
 
    // subscalar
    template <typename T>
    __global__ void subscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_subscalar(const T* a, const T scalar, T* c,const int size);
    

    // rsubscalar
    template <typename T>
    __global__ void rsubscalar_kernel(const T scalar, const T* A, T* C,const int size);

    template <typename T>
    void launch_rsubscalar(const T scalar, const T* a, T* c,const int size);

    // mul
    template <typename T>
    __global__ void mul_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_mul(const T* a, const T* b, T* c,const int size);
 
    // mulscalar
    template <typename T>
    __global__ void mulscalar_kernel(const T* A, const T scalar, T* C,const int size);  

    template <typename T>
    void launch_mulscalar(const T* a, const T scalar, T* c,const int size);

 
    // div
    template <typename T>
    __global__ void div_kernel(const T* A, const T* B, T* C,const int size);

    template <typename T>
    void launch_div(const T* a, const T* b, T* c,const int size);

 
    // divscalar
    template <typename T>
    __global__ void divscalar_kernel(const T* A, const T scalar, T* C,const int size);

    template <typename T>
    void launch_divscalar(const T* a, const T scalar, T* c,const int size);
 
    // rdivscalar
    template <typename T>
    __global__ void rdivscalar_kernel(const T scalar, const T* A, T* C,const int size);

    template <typename T>
    void launch_rdivscalar(const T scalar, const T* a, T* c,const int size);
 
    // invert
    template <typename T>
    __global__ void invert_kernel(const T* A, T* C,const int size);

    template <typename T>
    void launch_invert(const T* a, T* c,const int size);

}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
