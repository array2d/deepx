#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CU

#include <cuda_fp16.h>
#include <cuda_bf16.h>
 
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // sin
    template <typename T>
    __global__ void sin_kernel(const T* A, T* C, const int size);
    
    template <>
    __global__ void sin_kernel<double>(const double* A, double* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = sin(A[idx]);
        }
    }
    template <>
    __global__ void sin_kernel<float>(const float* A, float* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = sinf(A[idx]);
        }
    }

    template <>
    __global__ void sin_kernel<__half>(const __half* A, __half* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = hsin(A[idx]);
        }
    }   
    template <>
    __global__ void sin_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            C[idx] = hsin(A[idx]);
        }
    }

    template <typename T>
    void launch_sin(const T* a, T* c, const int size){
        auto [numBlocks, blockSize] = BestDims(size);
        sin_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch sin kernel",cudaGetLastError());
    }

    template void  launch_sin<double>(const double* a, double* c, const int size);
    template void  launch_sin<float>(const float* a, float* c, const int size);
    template void  launch_sin<__half>(const __half* a, __half* c, const int size);
    template void  launch_sin<nv_bfloat16>(const nv_bfloat16* a, nv_bfloat16* c, const int size);
    // cos
    template <typename T>
    __global__ void cos_kernel(const T* A, T* C, const int size);
    template <>
    __global__ void cos_kernel<double>(const double* A, double* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = cos(A[idx]);
        }
    }
    template <>
    __global__ void cos_kernel<float>(const float* A, float* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = cosf(A[idx]);
        }
    }   
 
    template <>
    __global__ void cos_kernel<__half>(const __half* A, __half* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = hcos(A[idx]);
        }
    }      
    template <>
    __global__ void cos_kernel<nv_bfloat16>(const nv_bfloat16 *A, nv_bfloat16 *C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = hcos(A[idx]);
        }
    }
    template <typename T>
    void launch_cos(const T* a, T* c, const int size){
        auto [numBlocks, blockSize] = BestDims(size);
        cos_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch cos kernel",cudaGetLastError());
    }
    template void  launch_cos<double>(const double* a, double* c, const int size);    
    template void  launch_cos<float>(const float* a, float* c, const int size);
    template void  launch_cos<__half>(const __half* a, __half* c, const int size);
    template void  launch_cos<nv_bfloat16>(const nv_bfloat16* a, nv_bfloat16* c, const int size);
    // tan
    template <typename T>
    __global__ void tan_kernel(const T* A, T* C, const int size);
    template <>
    __global__ void tan_kernel<double>(const double* A, double* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = tan(A[idx]);
        }
    }   
    template <>
    __global__ void tan_kernel<float>(const float* A, float* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = tanf(A[idx]);
        }
    }
   
 
    template <typename T>   
    void launch_tan(const T* a, T* c, const int size){
        auto [numBlocks, blockSize] = BestDims(size);
        tan_kernel<<<numBlocks, blockSize>>>(a, c, size);
        throwcudaerror("Failed to launch tan kernel",cudaGetLastError());
    }
        
    template void  launch_tan<double>( const double* a, double* c, const int size);
    template void  launch_tan<float>(const float* a, float* c, const int size);
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_SIN_CU
