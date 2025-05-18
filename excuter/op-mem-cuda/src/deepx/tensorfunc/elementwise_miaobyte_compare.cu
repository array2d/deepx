#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/vector_cuda.cuh"
namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void max_kernel(const T *A, const T *B, T *C, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            C[idx] = A[idx] > B[idx] ? A[idx] : B[idx];
        }
    }

    template <typename T>
    void launch_max(const T *A, const T *B, T *C, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        max_kernel<<<numBlocks, blockSize>>>(A, B, C, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_max<double>(const double *A, const double *B, double *C, const int size);
    template void launch_max<float>(const float *A, const float *B, float *C, const int size);
    template void launch_max<nv_bfloat16>(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, const int size);
    template void launch_max<__half>(const __half *A, const __half *B, __half *C, const int size);
    template void launch_max<int64_t>(const int64_t *A, const int64_t *B, int64_t *C, const int size);
    template void launch_max<int32_t>(const int32_t *A, const int32_t *B, int32_t *C, const int size);
    template void launch_max<int16_t>(const int16_t *A, const int16_t *B, int16_t *C, const int size);
    template void launch_max<int8_t>(const int8_t *A, const int8_t *B, int8_t *C, const int size);

    template <typename T>
    __global__ void maxscalar_kernel(const T *A, const T scalar, T *C, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            C[idx] = A[idx] > scalar ? A[idx] : scalar;
        }
    }

    template <typename T>
    void launch_maxscalar(const T *A, const T scalar, T *C, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        maxscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, C, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_maxscalar<double>(const double *A, const double scalar, double *C, const int size);
    template void launch_maxscalar<float>(const float *A, const float scalar, float *C, const int size);
    template void launch_maxscalar<nv_bfloat16>(const nv_bfloat16 *A, const nv_bfloat16 scalar, nv_bfloat16 *C, const int size);
    template void launch_maxscalar<__half>(const __half *A, const __half scalar, __half *C, const int size);
    template void launch_maxscalar<int64_t>(const int64_t *A, const int64_t scalar, int64_t *C, const int size);
    template void launch_maxscalar<int32_t>(const int32_t *A, const int32_t scalar, int32_t *C, const int size);
    template void launch_maxscalar<int16_t>(const int16_t *A, const int16_t scalar, int16_t *C, const int size);
    template void launch_maxscalar<int8_t>(const int8_t *A, const int8_t scalar, int8_t *C, const int size);

    template <typename T>
    __global__ void min_kernel(const T *A, const T *B, T *C, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            C[idx] = A[idx] < B[idx] ? A[idx] : B[idx];
        }
    }

    template <typename T>
    void launch_min(const T *A, const T *B, T *C, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        min_kernel<<<numBlocks, blockSize>>>(A, B, C, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_min<double>(const double *A, const double *B, double *C, const int size);
    template void launch_min<float>(const float *A, const float *B, float *C, const int size);
    template void launch_min<nv_bfloat16>(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, const int size);
    template void launch_min<__half>(const __half *A, const __half *B, __half *C, const int size);
    template void launch_min<int64_t>(const int64_t *A, const int64_t *B, int64_t *C, const int size);
    template void launch_min<int32_t>(const int32_t *A, const int32_t *B, int32_t *C, const int size);
    template void launch_min<int16_t>(const int16_t *A, const int16_t *B, int16_t *C, const int size);
    template void launch_min<int8_t>(const int8_t *A, const int8_t *B, int8_t *C, const int size);

    template <typename T>
    __global__ void minscalar_kernel(const T *A, const T scalar, T *C, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            C[idx] = A[idx] < scalar ? A[idx] : scalar;
        }
    }

    template <typename T>
    void launch_minscalar(const T *A, const T scalar, T *C, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        minscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, C, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_minscalar<double>(const double *A, const double scalar, double *C, const int size);
    template void launch_minscalar<float>(const float *A, const float scalar, float *C, const int size);
    template void launch_minscalar<nv_bfloat16>(const nv_bfloat16 *A, const nv_bfloat16 scalar, nv_bfloat16 *C, const int size);
    template void launch_minscalar<__half>(const __half *A, const __half scalar, __half *C, const int size);
    template void launch_minscalar<int64_t>(const int64_t *A, const int64_t scalar, int64_t *C, const int size);
    template void launch_minscalar<int32_t>(const int32_t *A, const int32_t scalar, int32_t *C, const int size);
    template void launch_minscalar<int16_t>(const int16_t *A, const int16_t scalar, int16_t *C, const int size);
    template void launch_minscalar<int8_t>(const int8_t *A, const int8_t scalar, int8_t *C, const int size);

    // equal
    template <typename T,typename MaskT>
    __global__ void equalwithepsilon_kernel(const T *A, const T *B, const float epsilon, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            float diff = fabsf(static_cast<float>(A[idx]) - static_cast<float>(B[idx]));
            if (diff < epsilon)
            {
                mask[idx] = 1;
            }
            else
            {
                mask[idx] = 0;
            }
        }
    }

    template <typename T,typename MaskT>
    __global__ void equal_kernel(const T *A, const T *B, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] == B[idx]);
        }
    }

    template <typename T,typename MaskT>
    void launch_equal(const T *A, const T *B, const float epsilon, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        if (epsilon == 0)
        {
            equal_kernel<<<numBlocks, blockSize>>>(A, B, mask, size);
        }
        else
        {
            equalwithepsilon_kernel<<<numBlocks, blockSize>>>(A, B, epsilon, mask, size);
        }
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_equal<double,bool>(const double *A, const double *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<float,bool>(const float *A, const float *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<__half,bool>(const __half *A, const __half *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<int64_t,bool>(const int64_t *A, const int64_t *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<int32_t,bool>(const int32_t *A, const int32_t *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<int16_t,bool>(const int16_t *A, const int16_t *B, const float epsilon, bool *mask, const int size);
    template void launch_equal<int8_t,bool>(const int8_t *A, const int8_t *B, const float epsilon, bool *mask, const int size);

    // equalscalar
    template <typename T,typename MaskT>
    __global__ void equalscalarwithepsilon_kernel(const T *A, const T scalar, const float epsilon, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {   
            float diff = fabsf(static_cast<float>(A[idx]) - static_cast<float>(scalar));
            if (diff < epsilon)
            {
                mask[idx] = 1;
            }
            else
            {
                mask[idx] = 0;
            }
        }
    }

    template <typename T,typename MaskT>
    __global__ void equalscalar_kernel(const T *A, const T scalar, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] == scalar);
        }
    }

    template <typename T,typename MaskT>
    void launch_equalscalar(const T *A, const T scalar, const float epsilon, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        if (epsilon == 0)
        {
            equalscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, mask, size);
        }
        else
        {
            equalscalarwithepsilon_kernel<<<numBlocks, blockSize>>>(A, scalar, epsilon, mask, size);
        }
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_equalscalar<double,bool>(const double *A, const double scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<float,bool>(const float *A, const float scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<__half,bool>(const __half *A, const __half scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<int64_t,bool>(const int64_t *A, const int64_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<int32_t,bool>(const int32_t *A, const int32_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<int16_t,bool>(const int16_t *A, const int16_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_equalscalar<int8_t,bool>(const int8_t *A, const int8_t scalar, const float epsilon, bool *mask, const int size);

    // not  equal
    template <typename T,typename MaskT>
    __global__ void notequalwithepsilon_kernel(const T *A, const T *B, const float epsilon, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            float diff = fabsf(static_cast<float>(A[idx]) - static_cast<float>(B[idx]));
            if (diff < epsilon)
            {
                mask[idx] = 0;
            }
            else
            {
                mask[idx] = 1;
            }
        }
    }

    template <typename T,typename MaskT>
    __global__ void notequal_kernel(const T *A, const T *B, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] != B[idx]);
        }
    }

    template <typename T,typename MaskT>
    void launch_notequal(const T *A, const T *B, const float epsilon, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        if (epsilon == 0)
        {
            notequal_kernel<<<numBlocks, blockSize>>>(A, B, mask, size);
        }
        else
        {
            notequalwithepsilon_kernel<<<numBlocks, blockSize>>>(A, B, epsilon, mask, size);
        }
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_notequal<double,bool>(const double *A, const double *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<float,bool>(const float *A, const float *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<__half,bool>(const __half *A, const __half *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<int64_t,bool>(const int64_t *A, const int64_t *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<int32_t,bool>(const int32_t *A, const int32_t *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<int16_t,bool>(const int16_t *A, const int16_t *B, const float epsilon, bool *mask, const int size);
    template void launch_notequal<int8_t,bool>(const int8_t *A, const int8_t *B, const float epsilon, bool *mask, const int size);

    // notequalscalar
    template <typename T,typename MaskT>
    __global__ void notequalscalarwithepsilon_kernel(const T *A, const T scalar, const float epsilon, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {   
            float diff = fabsf(static_cast<float>(A[idx]) - static_cast<float>(scalar));
            if (diff < epsilon)
            {
                mask[idx] = 0;
            }
            else
            {
                mask[idx] = 1;
            }
        }
    }

    template <typename T,typename MaskT>
    __global__ void notequalscalar_kernel(const T *A, const T scalar, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] != scalar);
        }
    }

    template <typename T,typename MaskT>
    void launch_notequalscalar(const T *A, const T scalar, const float epsilon, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        if (epsilon == 0)
        {
            notequalscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, mask, size);
        }
        else
        {
            notequalscalarwithepsilon_kernel<<<numBlocks, blockSize>>>(A, scalar, epsilon, mask, size);
        }
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_notequalscalar<double,bool>(const double *A, const double scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<float,bool>(const float *A, const float scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<__half,bool>(const __half *A, const __half scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<int64_t,bool>(const int64_t *A, const int64_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<int32_t,bool>(const int32_t *A, const int32_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<int16_t,bool>(const int16_t *A, const int16_t scalar, const float epsilon, bool *mask, const int size);
    template void launch_notequalscalar<int8_t,bool>(const int8_t *A, const int8_t scalar, const float epsilon, bool *mask, const int size);

    // less
    template <typename T,typename MaskT>
    __global__ void less_kernel(const T *A, const T *B, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] < B[idx]);
        }
    }

    template <typename T,typename MaskT>
    void launch_less(const T *A, const T *B, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        less_kernel<<<numBlocks, blockSize>>>(A, B, mask, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_less<double,bool>(const double *A, const double *B, bool *mask, const int size);
    template void launch_less<float,bool>(const float *A, const float *B, bool *mask, const int size);
    template void launch_less<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 *B, bool *mask, const int size);
    template void launch_less<__half,bool>(const __half *A, const __half *B, bool *mask, const int size);
    template void launch_less<int64_t,bool>(const int64_t *A, const int64_t *B, bool *mask, const int size);
    template void launch_less<int32_t,bool>(const int32_t *A, const int32_t *B, bool *mask, const int size);
    template void launch_less<int16_t,bool>(const int16_t *A, const int16_t *B, bool *mask, const int size);
    template void launch_less<int8_t,bool>(const int8_t *A, const int8_t *B, bool *mask, const int size);

    // lessscalar
    
    template <typename T,typename MaskT>
    __global__ void lessscalar_kernel(const T *A, const T scalar, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] < scalar);
        }
    }

    template <typename T,typename MaskT>
    void launch_lessscalar(const T *A, const T scalar, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        lessscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, mask, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_lessscalar<double,bool>(const double *A, const double scalar, bool *mask, const int size);
    template void launch_lessscalar<float,bool>(const float *A, const float scalar, bool *mask, const int size);
    template void launch_lessscalar<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 scalar, bool *mask, const int size);
    template void launch_lessscalar<__half,bool>(const __half *A, const __half scalar, bool *mask, const int size);
    template void launch_lessscalar<int64_t,bool>(const int64_t *A, const int64_t scalar, bool *mask, const int size);
    template void launch_lessscalar<int32_t,bool>(const int32_t *A, const int32_t scalar, bool *mask, const int size);
    template void launch_lessscalar<int16_t,bool>(const int16_t *A, const int16_t scalar, bool *mask, const int size);
    template void launch_lessscalar<int8_t,bool>(const int8_t *A, const int8_t scalar, bool *mask, const int size);
    
    // greater
    template <typename T,typename MaskT>
    __global__ void greater_kernel(const T *A, const T *B, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] > B[idx]);
        }
    }

    template <typename T,typename MaskT>
    void launch_greater(const T *A, const T *B, MaskT *mask, const int size)
    {   
        auto [numBlocks, blockSize] = BestDims(size);
        greater_kernel<<<numBlocks, blockSize>>>(A, B, mask, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }

    template void launch_greater<double,bool>(const double *A, const double *B, bool *mask, const int size);
    template void launch_greater<float,bool>(const float *A, const float *B, bool *mask, const int size);
    template void launch_greater<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 *B, bool *mask, const int size);
    template void launch_greater<__half,bool>(const __half *A, const __half *B, bool *mask, const int size);
    template void launch_greater<int64_t,bool>(const int64_t *A, const int64_t *B, bool *mask, const int size);
    template void launch_greater<int32_t,bool>(const int32_t *A, const int32_t *B, bool *mask, const int size);
    template void launch_greater<int16_t,bool>(const int16_t *A, const int16_t *B, bool *mask, const int size);
    template void launch_greater<int8_t,bool>(const int8_t *A, const int8_t *B, bool *mask, const int size);    

    // greaterscalar
    template <typename T,typename MaskT>
    __global__ void greaterscalar_kernel(const T *A, const T scalar, MaskT *mask, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            mask[idx] = (A[idx] > scalar);
        }
    }   

    template <typename T,typename MaskT>
    void launch_greaterscalar(const T *A, const T scalar, MaskT *mask, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        greaterscalar_kernel<<<numBlocks, blockSize>>>(A, scalar, mask, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }   

    template void launch_greaterscalar<double,bool>(const double *A, const double scalar, bool *mask, const int size);
    template void launch_greaterscalar<float,bool>(const float *A, const float scalar, bool *mask, const int size);
    template void launch_greaterscalar<nv_bfloat16,bool>(const nv_bfloat16 *A, const nv_bfloat16 scalar, bool *mask, const int size);
    template void launch_greaterscalar<__half,bool>(const __half *A, const __half scalar, bool *mask, const int size);
    template void launch_greaterscalar<int64_t,bool>(const int64_t *A, const int64_t scalar, bool *mask, const int size);
    template void launch_greaterscalar<int32_t,bool>(const int32_t *A, const int32_t scalar, bool *mask, const int size);
    template void launch_greaterscalar<int16_t,bool>(const int16_t *A, const int16_t scalar, bool *mask, const int size);
    template void launch_greaterscalar<int8_t,bool>(const int8_t *A, const int8_t scalar, bool *mask, const int size);

    // switch
    template <typename T,typename casesT>
    __global__ void switch_kernel(const T **tensorsdata, const int numTensors, const casesT *cases, T *C, const int size)
    {
        int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
        {
            C[idx] = tensorsdata[cases[idx]][idx];
        }
    }

    template <typename T,typename casesT>
    void launch_switch(const T **tensorsdata, const int numTensors, const casesT *cases, T *C, const int size)
    {
        auto [numBlocks, blockSize] = BestDims(size);
        cudaVector<const T *> tensorsdataList(tensorsdata, numTensors, cudaMemcpyHostToDevice);
        switch_kernel<<<numBlocks, blockSize>>>(tensorsdataList.data, numTensors, cases, C, size);
        throwcudaerror("Failed to launch add kernel",cudaGetLastError());
    }   
 
    
    template void launch_switch<double,int32_t>(const double **tensorsdata, const int numTensors, const int32_t *cases, double *C, const int size);
    template void launch_switch<float,int32_t>(const float **tensorsdata, const int numTensors, const int32_t *cases, float *C, const int size);
    template void launch_switch<nv_bfloat16,int32_t>(const nv_bfloat16 **tensorsdata, const int numTensors, const int32_t *cases, nv_bfloat16 *C, const int size);
    template void launch_switch<__half,int32_t>(const __half **tensorsdata, const int numTensors, const int32_t *cases, __half *C, const int size);
    template void launch_switch<int64_t,int32_t>(const int64_t **tensorsdata, const int numTensors, const int32_t *cases, int64_t *C, const int size);
    template void launch_switch<int32_t,int32_t>(const int32_t **tensorsdata, const int numTensors, const int32_t *cases, int32_t *C, const int size);
    template void launch_switch<int16_t,int32_t>(const int16_t **tensorsdata, const int numTensors, const int32_t *cases, int16_t *C, const int size);
    template void launch_switch<int8_t,int32_t>(const int8_t **tensorsdata, const int numTensors, const int32_t *cases, int8_t *C, const int size);
    template void launch_switch<bool,int32_t>(const bool **tensorsdata, const int numTensors, const int32_t *cases, bool *C, const int size);
    
    template void launch_switch<double,bool>(const double **tensorsdata, const int numTensors, const bool *cases, double *C, const int size);
    template void launch_switch<float,bool>(const float **tensorsdata, const int numTensors, const bool *cases, float *C, const int size);
    template void launch_switch<nv_bfloat16,bool>(const nv_bfloat16 **tensorsdata, const int numTensors, const bool *cases, nv_bfloat16 *C, const int size);
    template void launch_switch<__half,bool>(const __half **tensorsdata, const int numTensors, const bool *cases, __half *C, const int size);
    template void launch_switch<int64_t,bool>(const int64_t **tensorsdata, const int numTensors, const bool *cases, int64_t *C, const int size);
    template void launch_switch<int32_t,bool>(const int32_t **tensorsdata, const int numTensors, const bool *cases, int32_t *C, const int size);
    template void launch_switch<int16_t,bool>(const int16_t **tensorsdata, const int numTensors, const bool *cases, int16_t *C, const int size);
    template void launch_switch<int8_t,bool>(const int8_t **tensorsdata, const int numTensors, const bool *cases, int8_t *C, const int size);
    template void launch_switch<bool,bool>(const bool **tensorsdata, const int numTensors, const bool *cases, bool *C, const int size);
 
}
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_CU
