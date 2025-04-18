#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "deepx/shape_changeshape.hpp" //BroadcastMap类型

namespace deepx::tensorfunc
{
    // transpose
    template <typename T>
    __global__ void transpose_kernel(const T *input, const int *inputStrides, T *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <typename T>
    void launch_transpose(  const T *input, const int *inputStrides, T *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

   
    template <typename DIM, typename T>
    __global__ void concat_kernel(const T **tensorsData,
                                  const int *inputStrides,
                                  T *outputData,
                                  const int *outputStrides,
                                  const int dim,
                                  const int len,
                                  const int axis,
                                  const int numTensors,
                                  const int *shapeAtAxis);

    template <typename T>
    void launch_concat(const T **tensorsData, const int *inputStrides, T *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    

    __host__ __device__ void fromBroadcastIndices(const BroadcastMap *broadcastMap, const int *broadcastIndices, const int broadcastIndicesDim, int *indices);
    
    // broadcastTo
    template <int DIM, typename T>
    __global__ void broadcastTo_kernel(
        const T *input, const int *inputStrides,const int inputDim,
        const BroadcastMap *broadcastMap,
        T *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <typename T>
    void launch_broadcastTo(const T *input, const int *inputStrides,const int intputDim,
                            const BroadcastMap *broadcastMap,
                            T *output, const int *outputStrides,const int outputDim,const int outputlen);
 
};
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH