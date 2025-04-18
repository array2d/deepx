#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CUH

#include <cuda_bf16.h>  
#include <cuda_fp16.h>
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tensorfunc
{
    //max
    template <typename T>
    __global__ void max_kernel(const T* A, const T* B, T* C, const int size);

    template <typename T>
    void launch_max(const T* A, const T* B, T* C, const int size);
 
    //maxscalar
    template <typename T>
    __global__ void maxscalar_kernel(const T* A, const T scalar, T* C, const int size);

    template <typename T>
    void launch_maxscalar(const T* A, const T scalar, T* C, const int size);
    
    //min
    template <typename T>
    __global__ void min_kernel(const T* A, const T* B, T* C, const int size);

    template <typename T>
    void launch_min(const T* A, const T* B, T* C, const int size);

     
    //minscalar
    template <typename T>
    __global__ void minscalar_kernel(const T* A, const T scalar, T* C, const int size);

    template <typename T>
    void launch_minscalar(const T* A, const T scalar, T* C, const int size);
    
    
    //equal
    template <typename T,typename MaskT>
    __global__ void equal_kernel(const T* A, const T* B,const float epsilon, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    __global__ void equal_kernel(const T* A, const T* B, float* mask, const int size);
    
    template <typename T,typename MaskT>
    void launch_equal(const T* A, const T* B,const float epsilon, MaskT* mask, const int size);

    //equalscalar
    template <typename T,typename MaskT>
    __global__ void equalscalar_kernel(const T* A, const T scalar,const float epsilon, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    void launch_equalscalar(const T* A, const T scalar,const float epsilon, MaskT* mask, const int size);

    //less
    template <typename T,typename MaskT>
    __global__ void less_kernel(const T* A, const T* B, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    void launch_less(const T* A, const T* B, MaskT* mask, const int size);

    //lessscalar
    template <typename T,typename MaskT>
    __global__ void lessscalar_kernel(const T* A, const T scalar, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    void launch_lessscalar(const T* A, const T scalar, MaskT* mask, const int size);

    //greater
    template <typename T,typename MaskT>
    __global__ void greater_kernel(const T* A, const T* B, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    void launch_greater(const T* A, const T* B, MaskT* mask, const int size);

    //greaterscalar
    template <typename T,typename MaskT>
    __global__ void greaterscalar_kernel(const T* A, const T scalar, MaskT* mask, const int size);

    template <typename T,typename MaskT>
    void launch_greaterscalar(const T* A, const T scalar, MaskT* mask, const int size);
    
    //switch
    template <typename T,typename casesT>
    __global__ void switch_kernel(const T** tensorsdata,const int numTensors, const casesT* cases, T* C, const int size);

    template <typename T,typename casesT>
    void launch_switch(const T** tensorsdata,const int numTensors, const casesT* cases, T* C, const int size);
    
}
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CUH
