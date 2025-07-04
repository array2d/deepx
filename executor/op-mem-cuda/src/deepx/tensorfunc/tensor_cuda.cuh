#ifndef DEEPX_TENSORFUNC_TENSOR_CUDA_CUH
#define DEEPX_TENSORFUNC_TENSOR_CUDA_CUH

#include <cuda_runtime.h>
#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    inline __host__ __device__ void linearTo(const int *strides, const int dim, int *indices, const int id)
    {
        int linearIndex = id;
        for (int i = 0; i < dim; i++)
        {
            indices[i] = linearIndex / strides[i];
            linearIndex %= strides[i];
        }
    }

    inline __host__ __device__ int linearAt(const int *strides, const int dim,const int *indices)
    {
        int idx = 0;
        for (int i = 0; i < dim; i++)
        {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    template <typename T>
    __device__ __host__ void reorder(const T *order, const int *dimOrder, int dim, T *neworder)
    {
        for (int i = 0; i < dim; i++)
        {
            neworder[i] = order[dimOrder[i]];
        }
    }
    
    const int MAX_DIM = 12;
}

#endif // DEEPX_TENSORFUNC_TENSOR_CUDA_CUH
