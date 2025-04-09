#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "deepx/shape_changeshape.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // transpose
    template <typename T>
    __global__ void transpose_kernel(const T *input, const int *inputStrides, T *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <typename T>
    void launch_transpose(const int numBlocks, const int blockSize, const T *input, const int *inputStrides, T *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<double>(const int numBlocks, const int blockSize, const double *input, const int *inputStrides, double *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<float>(const int numBlocks, const int blockSize, const float *input, const int *inputStrides, float *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16 *input, const int *inputStrides, nv_bfloat16 *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<__half>(const int numBlocks, const int blockSize, const __half *input, const int *inputStrides, __half *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<int64_t>(const int numBlocks, const int blockSize, const int64_t *input, const int *inputStrides, int64_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<int32_t>(const int numBlocks, const int blockSize, const int32_t *input, const int *inputStrides, int32_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<int16_t>(const int numBlocks, const int blockSize, const int16_t *input, const int *inputStrides, int16_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    template <>
    void launch_transpose<int8_t>(const int numBlocks, const int blockSize, const int8_t *input, const int *inputStrides, int8_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

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

    template <>
    void launch_concat<double>(const double **tensorsData, const int *inputStrides, double *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<float>(const float **tensorsData, const int *inputStrides, float *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<nv_bfloat16>(const nv_bfloat16 **tensorsData, const int *inputStrides, nv_bfloat16 *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<__half>(const __half **tensorsData, const int *inputStrides, __half *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<int64_t>(const int64_t **tensorsData, const int *inputStrides, int64_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<int32_t>(const int32_t **tensorsData, const int *inputStrides, int32_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<int16_t>(const int16_t **tensorsData, const int *inputStrides, int16_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    template <>
    void launch_concat<int8_t>(const int8_t **tensorsData, const int *inputStrides, int8_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);


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

    template <>
    void launch_broadcastTo<double>(const double *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    double *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<float>(const float *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    float *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<nv_bfloat16>(const nv_bfloat16 *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    nv_bfloat16 *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<__half>(const __half *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    __half *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<int64_t>(const int64_t *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    int64_t *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<int32_t>(const int32_t *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    int32_t *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<int16_t>(const int16_t *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    int16_t *output, const int *outputStrides,const int outputDim,const int outputlen);

    template <>
    void launch_broadcastTo<int8_t>(const int8_t *input, const int *inputStrides,const int inputDim,
                                    const BroadcastMap *broadcastMap,
                                    int8_t *output, const int *outputStrides,const int outputDim,const int outputlen);
}
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP