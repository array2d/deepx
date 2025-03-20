#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
     template <typename T>
    __global__ void add_kernel(const T* A, const T* B, T* C,const int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] + B[idx];
        }
    }
    template __global__ void add_kernel<double>(const double* A, const double* B, double* C,const int size);
    template __global__ void add_kernel<float>(const float* A, const float* B, float* C,const int size);
    template __global__ void add_kernel<half>(const half* A, const half* B, half* C,const int size);
    template __global__ void add_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C,const int size);
    template __global__ void add_kernel<int64_t>(const int64_t* A, const int64_t* B, int64_t* C,const int size);
    template __global__ void add_kernel<int32_t>(const int32_t* A, const int32_t* B, int32_t* C,const int size);
    template __global__ void add_kernel<int16_t>(const int16_t* A, const int16_t* B, int16_t* C,const int size);
    template __global__ void add_kernel<int8_t>(const int8_t* A, const int8_t* B, int8_t* C,const int size);
    

    template <typename T>
    void launch_add(int numBlocks, int blockSize,const T*  a, const  T* b,  T* c,const int size)
    {
         // 启动kernel
            add_kernel<<<numBlocks, blockSize>>>(a, b, c, size);
            // 检查kernel执行是否成功
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }

    template void launch_add<double>(int numBlocks, int blockSize,const double*  a, const  double* b,  double* c,const int size);
    template void launch_add<float>(int numBlocks, int blockSize,const float*  a, const  float* b,  float* c,const int size);
    template void launch_add<half>(int numBlocks, int blockSize,const half*  a, const  half* b,  half* c,const int size);
    template void launch_add<nv_bfloat16>(int numBlocks, int blockSize,const nv_bfloat16*  a, const  nv_bfloat16* b,  nv_bfloat16* c,const int size);
    template void launch_add<int64_t>(int numBlocks, int blockSize,const int64_t*  a, const  int64_t* b,  int64_t* c,const int size);
    template void launch_add<int32_t>(int numBlocks, int blockSize, const int32_t*  a, const  int32_t* b,  int32_t* c,const int size);
    template void launch_add<int16_t>(int numBlocks, int blockSize, const int16_t*  a, const  int16_t* b,  int16_t* c,const int size);
    template void launch_add<int8_t>(int numBlocks, int blockSize, const int8_t*  a, const  int8_t* b,  int8_t* c,const int size);


    template <typename T>
    __global__ void addscalar_kernel(const T* A, const T scalar, T* C,const int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] + scalar;
        }
    }   
    template __global__ void addscalar_kernel<double>(const double* A, const double scalar, double* C,const int size);   
    template __global__ void addscalar_kernel<float>(const float* A, const float scalar, float* C,const int size);
    template __global__ void addscalar_kernel<half>(const half* A, const half scalar, half* C,const int size);
    template __global__ void addscalar_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C,const int size);
    template __global__ void addscalar_kernel<int64_t>(const int64_t* A, const int64_t scalar, int64_t* C,const int size);
    template __global__ void addscalar_kernel<int32_t>(const int32_t* A, const int32_t scalar, int32_t* C,const int size);
    template __global__ void addscalar_kernel<int16_t>(const int16_t* A, const int16_t scalar, int16_t* C,const int size);
    template __global__ void addscalar_kernel<int8_t>(const int8_t* A, const int8_t scalar, int8_t* C,const int size);
    
    template <typename T>
    void launch_addscalar(const int numBlocks, const int blockSize, const T* a, const T scalar, T* c, const int size) {
        addscalar_kernel<<<numBlocks, blockSize>>>(a, scalar, c, size);
    }   
    template void launch_addscalar<double>(const int numBlocks, const int blockSize, const double* a, const double scalar, double* c, const int size);
    template void launch_addscalar<float>(const int numBlocks, const int blockSize, const float* a, const float scalar, float* c, const int size);
    template void launch_addscalar<half>(const int numBlocks, const int blockSize, const half* a, const half scalar, half* c, const int size);
    template void launch_addscalar<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16* a, const nv_bfloat16 scalar, nv_bfloat16* c, const int size);
    template void launch_addscalar<int64_t>(const int numBlocks, const int blockSize, const int64_t* a, const int64_t scalar, int64_t* c, const int size);  
    template void launch_addscalar<int32_t>(const int numBlocks, const int blockSize, const int32_t* a, const int32_t scalar, int32_t* c, const int size);
    template void launch_addscalar<int16_t>(const int numBlocks, const int blockSize, const int16_t* a, const int16_t scalar, int16_t* c, const int size);
    template void launch_addscalar<int8_t>(const int numBlocks, const int blockSize, const int8_t* a, const int8_t scalar, int8_t* c, const int size);


    template <typename T>
    __global__ void sub_kernel(const T* A, const T* B, T* C,const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] - B[idx];
        }   
    }
    template __global__ void sub_kernel<double>(const double* A, const double* B, double* C, const int size);   
    template __global__ void sub_kernel<float>(const float* A, const float* B, float* C, const int size);
    template __global__ void sub_kernel<half>(const half* A, const half* B, half* C, const int size);
    template __global__ void sub_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int size);
    template __global__ void sub_kernel<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, const int size);
    template __global__ void sub_kernel<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, const int size);
    template __global__ void sub_kernel<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, const int size);
    template __global__ void sub_kernel<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, const int size);

    template <typename T>
    void launch_sub(const int numBlocks, const int blockSize, const T* a, const T* b, T* c, const int size) {
        sub_kernel<<<numBlocks, blockSize>>>(a, b, c, size);
    }
    template void launch_sub<double>(const int numBlocks, const int blockSize, const double* a, const double* b, double* c, const int size);
    template void launch_sub<float>(const int numBlocks, const int blockSize, const float* a, const float* b, float* c, const int size);
    template void launch_sub<half>(const int numBlocks, const int blockSize, const half* a, const half* b, half* c, const int size);
    template void launch_sub<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16* a, const nv_bfloat16* b, nv_bfloat16* c, const int size);
    template void launch_sub<int64_t>(const int numBlocks, const int blockSize, const int64_t* a, const int64_t* b, int64_t* c, const int size);
    template void launch_sub<int32_t>(const int numBlocks, const int blockSize, const int32_t* a, const int32_t* b, int32_t* c, const int size);
    template void launch_sub<int16_t>(const int numBlocks, const int blockSize, const int16_t* a, const int16_t* b, int16_t* c, const int size);
    template void launch_sub<int8_t>(const int numBlocks, const int blockSize, const int8_t* a, const int8_t* b, int8_t* c, const int size);    
    
    template <typename T>
    __global__ void subscalar_kernel(const T* A, const T scalar, T* C,const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] - scalar;
        }
    }   
    template __global__ void subscalar_kernel<double>(const double* A, const double scalar, double* C,const int size);
    template __global__ void subscalar_kernel<float>(const float* A, const float scalar, float* C,const int size);
    template __global__ void subscalar_kernel<half>(const half* A, const half scalar, half* C,const int size);
    template __global__ void subscalar_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C,const int size);
    template __global__ void subscalar_kernel<int64_t>(const int64_t* A, const int64_t scalar, int64_t* C,const int size);  
    template __global__ void subscalar_kernel<int32_t>(const int32_t* A, const int32_t scalar, int32_t* C,const int size);  
    template __global__ void subscalar_kernel<int16_t>(const int16_t* A, const int16_t scalar, int16_t* C,const int size);  
    template __global__ void subscalar_kernel<int8_t>(const int8_t* A, const int8_t scalar, int8_t* C,const int size);  

    template <typename T>
    void launch_subscalar(const int numBlocks, const int blockSize, const T* a, const T scalar, T* c, const int size) { 
        subscalar_kernel<<<numBlocks, blockSize>>>(a, scalar, c, size);
    }
    template void launch_subscalar<double>(const int numBlocks, const int blockSize, const double* a, const double scalar, double* c, const int size);
    template void launch_subscalar<float>(const int numBlocks, const int blockSize, const float* a, const float scalar, float* c, const int size);
    template void launch_subscalar<half>(const int numBlocks, const int blockSize, const half* a, const half scalar, half* c, const int size);
    template void launch_subscalar<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16* a, const nv_bfloat16 scalar, nv_bfloat16* c, const int size);  
    template void launch_subscalar<int64_t>(const int numBlocks, const int blockSize, const int64_t* a, const int64_t scalar, int64_t* c, const int size);  
    template void launch_subscalar<int32_t>(const int numBlocks, const int blockSize, const int32_t* a, const int32_t scalar, int32_t* c, const int size);  
    template void launch_subscalar<int16_t>(const int numBlocks, const int blockSize, const int16_t* a, const int16_t scalar, int16_t* c, const int size);  
    template void launch_subscalar<int8_t>(const int numBlocks, const int blockSize, const int8_t* a, const int8_t scalar, int8_t* c, const int size);    
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
