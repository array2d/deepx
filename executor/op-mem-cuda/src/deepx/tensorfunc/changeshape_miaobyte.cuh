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
    void launch_transpose(const T *input, const int *inputStrides, T *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

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

    // broadcastTo
    __host__ __device__ void fromBroadcastIndices(const BroadcastMap *broadcastMap, const int *broadcastIndices, const int broadcastIndicesDim, int *indices);

    template <int DIM, typename T>
    __global__ void broadcastTo_kernel(
        const T *input, const int *inputStrides, const int inputDim,
        const BroadcastMap *broadcastMap,
        T *output, const int *outputStrides, const int outputDim, const int outputlen);

    template <typename T>
    void launch_broadcastTo(const T *input, const int *inputStrides, const int intputDim,
                            const BroadcastMap *broadcastMap,
                            T *output, const int *outputStrides, const int outputDim, const int outputlen);

    // indexselect
     template <typename GatherAxisT>
    __host__ __device__ void fromIndexselectIndices(
    const int *output_indices,const int outputDim,  // 输出张量的索引
    const GatherAxisT *indices,const int *indicesStrides,const int indicesDim, //indices是tensor
    int *index_indices,
    const int gatherAxis,      // gather操作的轴
    int *input_indices,const int inputDim);       // 计算出的输入张量索引  

    template <int DIM, typename T,typename GatherAxisT>
    __global__ void indexselect_kernel(
        const T *input, const int *inputStrides, const int inputDim,
        const GatherAxisT *index,const int *indexStrides,const int indexDim,
        const int gatherAxis,
        T *output,const int *outputStrides,const int outputDim,const int outputlen);

    template <typename T,typename GatherAxisT>
    void launch_indexselect(
        const T *input, const int *inputStrides, const int inputDim, 
        const GatherAxisT *indices,const int *indicesStrides,const int indicesDim,
        const int gatherAxis,
        T *output,const int *outputStrides,const int outputDim,const int outputlen);


    // repeat
    template <int DIM, typename T>
    __global__ void repeat_kernel(
        const T *input, const int *inputStrides, 
        const int *repeats,
        T *output, const int *outputStrides, const int outputlen,
        const int dim);
 
    template <typename T>
    void launch_repeat(
        const T *input, const int *inputStrides, 
        const int *repeats, 
        T *output, const int *outputStrides, const int outputlen,
        const int dim);

    // repeat_interleave
    template <int DIM, typename T>
    __global__ void repeat_interleave_kernel(
        const T *input, const int *inputStrides,
        const int *repeats,
        T *output, const int *outputStrides, const int outputlen,
        const int dim);
};
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH