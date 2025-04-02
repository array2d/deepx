#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void max_kernel(const T* A, const T* B, T* C, const int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] > B[idx] ? A[idx] : B[idx];
        }
    }

    template __global__ void max_kernel<double>(const double* A, const double* B, double* C, const int size);
    template __global__ void max_kernel<float>(const float* A, const float* B, float* C, const int size);
    template __global__ void max_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int size);
    template __global__ void max_kernel<__half>(const __half* A, const __half* B, __half* C, const int size);
    template __global__ void max_kernel<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, const int size);
    template __global__ void max_kernel<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, const int size);
    template __global__ void max_kernel<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, const int size);
    template __global__ void max_kernel<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, const int size);

    template <typename T>
    void launch_max(int numBlocks, int blockSize, const T* A, const T* B, T* C, const int size)
    {
        max_kernel<<<numBlocks, blockSize>>>(A, B, C, size);
         cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }

    template void launch_max<double>(int numBlocks, int blockSize, const double* A, const double* B, double* C, const int size);
    template void launch_max<float>(int numBlocks, int blockSize, const float* A, const float* B, float* C, const int size);
    template void launch_max<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int size);
    template void launch_max<__half>(int numBlocks, int blockSize, const __half* A, const __half* B, __half* C, const int size);
    template void launch_max<int64_t>(int numBlocks, int blockSize, const int64_t* A, const int64_t* B, int64_t* C, const int size);
    template void launch_max<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t* B, int32_t* C, const int size);
    template void launch_max<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t* B, int16_t* C, const int size);
    template void launch_max<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t* B, int8_t* C, const int size);

    template <typename T>
    __global__ void maxscalar_kernel(const T* A, const T scalar, T* C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] > scalar ? A[idx] : scalar;
        }
    }

    template __global__ void maxscalar_kernel<double>(const double* A, const double scalar, double* C, const int size);
    template __global__ void maxscalar_kernel<float>(const float* A, const float scalar, float* C, const int size);
    template __global__ void maxscalar_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C, const int size);
    template __global__ void maxscalar_kernel<__half>(const __half* A, const __half scalar, __half* C, const int size);
    template __global__ void maxscalar_kernel<int64_t>(const int64_t* A, const int64_t scalar, int64_t* C, const int size);
    template __global__ void maxscalar_kernel<int32_t>(const int32_t* A, const int32_t scalar, int32_t* C, const int size); 
    template __global__ void maxscalar_kernel<int16_t>(const int16_t* A, const int16_t scalar, int16_t* C, const int size);
    template __global__ void maxscalar_kernel<int8_t>(const int8_t* A, const int8_t scalar, int8_t* C, const int size);

    template <typename T>   
    void launch_maxscalar(int numBlocks, int blockSize, const T* A, const T scalar, T* C, const int size)
    {
        maxscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, C, size);
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }

    template void launch_maxscalar<double>(int numBlocks, int blockSize, const double* A, const double scalar, double* C, const int size);
    template void launch_maxscalar<float>(int numBlocks, int blockSize, const float* A, const float scalar, float* C, const int size);
    template void launch_maxscalar<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C, const int size);
    template void launch_maxscalar<__half>(int numBlocks, int blockSize, const __half* A, const __half scalar, __half* C, const int size);
    template void launch_maxscalar<int64_t>(int numBlocks, int blockSize,   const int64_t* A, const int64_t scalar, int64_t* C, const int size);
    template void launch_maxscalar<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t scalar, int32_t* C, const int size);
    template void launch_maxscalar<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t scalar, int16_t* C, const int size);
    template void launch_maxscalar<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t scalar, int8_t* C, const int size);

    template <typename T>
    __global__ void min_kernel(const T* A, const T* B, T* C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] < B[idx] ? A[idx] : B[idx];
        }   
    }

    template __global__ void min_kernel<double>(const double* A, const double* B, double* C, const int size);
    template __global__ void min_kernel<float>(const float* A, const float* B, float* C, const int size);
    template __global__ void min_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int size);
    template __global__ void min_kernel<__half>(const __half* A, const __half* B, __half* C, const int size);
    template __global__ void min_kernel<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, const int size);
    template __global__ void min_kernel<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, const int size);
    template __global__ void min_kernel<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, const int size);
    template __global__ void min_kernel<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, const int size);

    template <typename T>
    void launch_min(int numBlocks, int blockSize, const T* A, const T* B, T* C, const int size)
    {
        min_kernel<<<numBlocks, blockSize>>>(A, B, C, size);
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }   
    
    template void launch_min<double>(int numBlocks, int blockSize, const double* A, const double* B, double* C, const int size);
    template void launch_min<float>(int numBlocks, int blockSize, const float* A, const float* B, float* C, const int size);
    template void launch_min<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int size);
    template void launch_min<__half>(int numBlocks, int blockSize, const __half* A, const __half* B, __half* C, const int size);
    template void launch_min<int64_t>(int numBlocks, int blockSize, const int64_t* A, const int64_t* B, int64_t* C, const int size);
    template void launch_min<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t* B, int32_t* C, const int size);
    template void launch_min<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t* B, int16_t* C, const int size);
    template void launch_min<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t* B, int8_t* C, const int size);

    template <typename T>
    __global__ void minscalar_kernel(const T* A, const T scalar, T* C, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] < scalar ? A[idx] : scalar;
        }
    }

    template __global__ void minscalar_kernel<double>(const double* A, const double scalar, double* C, const int size);
    template __global__ void minscalar_kernel<float>(const float* A, const float scalar, float* C, const int size);
    template __global__ void minscalar_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C, const int size); 
    template __global__ void minscalar_kernel<__half>(const __half* A, const __half scalar, __half* C, const int size);
    template __global__ void minscalar_kernel<int64_t>(const int64_t* A, const int64_t scalar, int64_t* C, const int size);
    template __global__ void minscalar_kernel<int32_t>(const int32_t* A, const int32_t scalar, int32_t* C, const int size);
    template __global__ void minscalar_kernel<int16_t>(const int16_t* A, const int16_t scalar, int16_t* C, const int size);
    template __global__ void minscalar_kernel<int8_t>(const int8_t* A, const int8_t scalar, int8_t* C, const int size);

    template <typename T>
    void launch_minscalar(int numBlocks, int blockSize, const T* A, const T scalar, T* C, const int size)
    {
        minscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, C, size);
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }   

    template void launch_minscalar<double>(int numBlocks, int blockSize, const double* A, const double scalar, double* C, const int size);
    template void launch_minscalar<float>(int numBlocks, int blockSize, const float* A, const float scalar, float* C, const int size);
    template void launch_minscalar<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16 scalar, nv_bfloat16* C, const int size);
    template void launch_minscalar<__half>(int numBlocks, int blockSize, const __half* A, const __half scalar, __half* C, const int size);
    template void launch_minscalar<int64_t>(int numBlocks, int blockSize, const int64_t* A, const int64_t scalar, int64_t* C, const int size);
    template void launch_minscalar<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t scalar, int32_t* C, const int size);
    template void launch_minscalar<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t scalar, int16_t* C, const int size);
    template void launch_minscalar<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t scalar, int8_t* C, const int size);  

    template <typename T>
    __global__ void compare_kernel(const T* A, const T* B, float* mask, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            if (A[idx] == B[idx]) {
                mask[idx] = 0.5;
            } else if (A[idx] > B[idx]) {
                mask[idx] = 1;
            } else {
                mask[idx] = 0;
            }
        }
    }

    template __global__ void compare_kernel<double>(const double* A, const double* B, float* mask, const int size);
    template __global__ void compare_kernel<float>(const float* A, const float* B, float* mask, const int size);
    template __global__ void compare_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, float* mask, const int size);    
    template __global__ void compare_kernel<__half>(const __half* A, const __half* B, float* mask, const int size);
    template __global__ void compare_kernel<int64_t>(const int64_t* A, const int64_t* B, float* mask, const int size);
    template __global__ void compare_kernel<int32_t>(const int32_t* A, const int32_t* B, float* mask, const int size);
    template __global__ void compare_kernel<int16_t>(const int16_t* A, const int16_t* B, float* mask, const int size);
    template __global__ void compare_kernel<int8_t>(const int8_t* A, const int8_t* B, float* mask, const int size);

    template <typename T>
    void launch_compare(int numBlocks, int blockSize, const T* A, const T* B, float* mask, const int size)
    {
        compare_kernel<<<numBlocks, blockSize>>>(A, B, mask, size);
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }

    template void launch_compare<double>(int numBlocks, int blockSize, const double* A, const double* B, float* mask, const int size);
    template void launch_compare<float>(int numBlocks, int blockSize, const float* A, const float* B, float* mask, const int size);
    template void launch_compare<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16* B, float* mask, const int size);
    template void launch_compare<__half>(int numBlocks, int blockSize, const __half* A, const __half* B, float* mask, const int size);
    template void launch_compare<int64_t>(int numBlocks, int blockSize, const int64_t* A, const int64_t* B, float* mask, const int size);
    template void launch_compare<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t* B, float* mask, const int size);
    template void launch_compare<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t* B, float* mask, const int size);
    template void launch_compare<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t* B, float* mask, const int size);
    
    //comparescalar
    template <typename T>
    __global__ void comparescalar_kernel(const T* A, const T scalar, float* mask, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {   
            if (A[idx] == scalar) {
                mask[idx] = 0.5;
            } else if (A[idx] > scalar) {
                mask[idx] = 1;
            } else {
                mask[idx] = 0;
            }
        }
    }

    template __global__ void comparescalar_kernel<double>(const double* A, const double scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<float>(const float* A, const float scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16 scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<__half>(const __half* A, const __half scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<int64_t>(const int64_t* A, const int64_t scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<int32_t>(const int32_t* A, const int32_t scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<int16_t>(const int16_t* A, const int16_t scalar, float* mask, const int size);
    template __global__ void comparescalar_kernel<int8_t>(const int8_t* A, const int8_t scalar, float* mask, const int size);

    template <typename T>
    void launch_comparescalar(int numBlocks, int blockSize, const T* A, const T scalar, float* mask, const int size)
    {
        comparescalar_kernel<<<numBlocks, blockSize>>>(A, scalar, mask, size);
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }   

    template void launch_comparescalar<double>(int numBlocks, int blockSize, const double* A, const double scalar, float* mask, const int size);
    template void launch_comparescalar<float>(int numBlocks, int blockSize, const float* A, const float scalar, float* mask, const int size);
    template void launch_comparescalar<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* A, const nv_bfloat16 scalar, float* mask, const int size);
    template void launch_comparescalar<__half>(int numBlocks, int blockSize, const __half* A, const __half scalar, float* mask, const int size);
    template void launch_comparescalar<int64_t>(int numBlocks, int blockSize, const int64_t* A, const int64_t scalar, float* mask, const int size);
    template void launch_comparescalar<int32_t>(int numBlocks, int blockSize, const int32_t* A, const int32_t scalar, float* mask, const int size);
    template void launch_comparescalar<int16_t>(int numBlocks, int blockSize, const int16_t* A, const int16_t scalar, float* mask, const int size);
    template void launch_comparescalar<int8_t>(int numBlocks, int blockSize, const int8_t* A, const int8_t scalar, float* mask, const int size);

};
    
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU
