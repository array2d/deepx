#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CU
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_CU

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/tensor_cuda.cuh"
#include "deepx/tensorfunc/vector_cuda.cuh"
#include "deepx/shape_changeshape.hpp"

namespace deepx::tensorfunc
{
    // transpose
    template <int DIM, typename T>
    __global__ void transpose_kernel(const T *inputData,
                                     const int *inputStrides,
                                     T *outputData,
                                     const int *outputStrides,
                                     const int dim,
                                     const int len,
                                     const int *dimOrder)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < len; thread_id += grid_stride)
        {
            int input_indices[DIM];

            // 计算当前线程需要处理的索引
            linearTo(inputStrides, dim, input_indices, thread_id);

            int output_indices[DIM];

            // 根据 dim_order 和输入输出的形状计算新索引
            reorder(input_indices, dimOrder, dim, output_indices);
            int inputIdx = linearAt(inputStrides, dim, input_indices);
            int outputIdx = linearAt(outputStrides, dim, output_indices);
            outputData[outputIdx] = inputData[inputIdx];
        }
    }

    template <typename T>
    void launch_transpose(const T *input,
                          const int *inputStrides,
                          T *output,
                          const int *outputStrides,
                          const int dim,
                          const int len,
                          const int *dimOrder)
    {
        cudaVector<int> strides_d(inputStrides, dim);
        cudaVector<int> newStrides_d(outputStrides, dim);
        cudaVector<int> dimOrder_d(dimOrder, dim);

        auto [numBlocks, blockSize] = BestDims(len);
        switch (dim)
        {
        case 1:
            transpose_kernel<1, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 2:
            transpose_kernel<2, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 3:
            transpose_kernel<3, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 4:
            transpose_kernel<4, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 5:
            transpose_kernel<5, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 6:
            transpose_kernel<6, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 7:
            transpose_kernel<7, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 8:
            transpose_kernel<8, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 9:
            transpose_kernel<9, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 10:
            transpose_kernel<10, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 11:
            transpose_kernel<11, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 12:
            transpose_kernel<12, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;

        default:
            throw std::runtime_error("dimension large than " + std::to_string(MAX_DIM));
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cuda error");
        }
    }

    template void launch_transpose<double>(const double *input, const int *inputStrides, double *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<float>(const float *input, const int *inputStrides, float *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<nv_bfloat16>(const nv_bfloat16 *input, const int *inputStrides, nv_bfloat16 *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<__half>(const __half *input, const int *inputStrides, __half *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int64_t>(const int64_t *input, const int *inputStrides, int64_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int32_t>(const int32_t *input, const int *inputStrides, int32_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int16_t>(const int16_t *input, const int *inputStrides, int16_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int8_t>(const int8_t *input, const int *inputStrides, int8_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

    // concat
    template <int DIM, typename T>
    __global__ void concat_kernel(const T **tensorsData,
                                  const int *inputStrides,
                                  T *outputData,
                                  const int *outputStrides,
                                  const int dim,
                                  const int outputLen,
                                  const int axis,
                                  const int numTensors,
                                  const int *shapeAtAxis)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        cudaVector<int> outputIndices(DIM);
        cudaVector<int> currentTensorIndices(DIM);
        for (; thread_id < outputLen; thread_id += grid_stride)
        {
            linearTo(outputStrides, dim, outputIndices.data, thread_id);
            int concatIdxResult = outputIndices[axis];
            int concatIdxCurrentTensor = concatIdxResult;
            int tensorIdx = 0;
            while (tensorIdx < numTensors)
            {
                if (concatIdxCurrentTensor < shapeAtAxis[tensorIdx])
                {
                    break;
                }
                else
                {
                    concatIdxCurrentTensor -= shapeAtAxis[tensorIdx];
                    tensorIdx++;
                }
            }
            currentTensorIndices.copyFromDevice(outputIndices.data, dim);
            currentTensorIndices[axis] = concatIdxCurrentTensor;

            int idxCurrentTensor = linearAt(inputStrides + tensorIdx * dim, dim, currentTensorIndices.data);

            int idx = linearAt(outputStrides, dim, outputIndices.data);
            outputData[idx] = tensorsData[tensorIdx][idxCurrentTensor];
        }
    }

    template <typename T>
    void launch_concat(
        const T **tensorsData,
        const int *inputStrides,
        T *outputData,
        const int *outputStrides,
        const int dim,
        const int outputLen,
        const int axis,
        const int numTensors,
        const int *shapeAtAxis)
    {
        auto [numBlocks, blockSize] = BestDims(outputLen);

        // output
        cudaVector<int> outputStrides_d(outputStrides, dim, cudaMemcpyHostToDevice);

        // input
        // datas
        cudaVector<const T *> tensorsDataList(tensorsData, numTensors, cudaMemcpyHostToDevice);
        // strides
        cudaVector<int> inputStrides_d(inputStrides, numTensors * dim, cudaMemcpyHostToDevice);

        // shapeAtAxis
        cudaVector<int> shapeAtAxis_d(shapeAtAxis, numTensors, cudaMemcpyHostToDevice);
        switch (dim)
        {
        case 1:
            concat_kernel<1, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 2:
            concat_kernel<2, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 3:
            concat_kernel<3, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 4:
            concat_kernel<4, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 5:
            concat_kernel<5, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 6:
            concat_kernel<6, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 7:
            concat_kernel<7, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 8:
            concat_kernel<8, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 9:
            concat_kernel<9, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 10:
            concat_kernel<10, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 11:
            concat_kernel<11, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 12:
            concat_kernel<12, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;

        default:
            throw std::runtime_error("dimension large than " + std::to_string(MAX_DIM));
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cuda error");
        }
    }
    template void launch_concat<double>(const double **tensorsData, const int *inputStrides, double *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<float>(const float **tensorsData, const int *inputStrides, float *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<nv_bfloat16>(const nv_bfloat16 **tensorsData, const int *inputStrides, nv_bfloat16 *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<__half>(const __half **tensorsData, const int *inputStrides, __half *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<int64_t>(const int64_t **tensorsData, const int *inputStrides, int64_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<int32_t>(const int32_t **tensorsData, const int *inputStrides, int32_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<int16_t>(const int16_t **tensorsData, const int *inputStrides, int16_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);
    template void launch_concat<int8_t>(const int8_t **tensorsData, const int *inputStrides, int8_t *outputData, const int *outputStrides, const int dim, const int len, const int axis, const int numTensors, const int *shapeAtAxis);

    // broadcastTo
    __host__ __device__ void fromBroadcastIndices(const BroadcastMap *broadcastMap, const int *broadcastIndices, const int broadcastIndicesDim, int *indices)
    {
        for (int i = 0, j = 0; i < broadcastIndicesDim; ++i)
        {
            switch (broadcastMap[i])
            {
            case xTox:
                indices[j++] = broadcastIndices[i];
                break;
            case nullTo1:
                break;
            case xTo1:
                indices[j++] = 0;
                break;
            }
        }
    }

    template <int DIM, typename T>
    __global__ void broadcastTo_kernel(const T *input, const int *inputStrides, const int inputDim,
                                       const BroadcastMap *broadcastMap,
                                       T *output, const int *outputStrides, const int outputDim, const int outputlen)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < outputlen; thread_id += grid_stride)
        {
            int output_indices[DIM];
            linearTo(outputStrides, outputDim, output_indices, thread_id);
            int input_indices[DIM];
            fromBroadcastIndices(broadcastMap, output_indices, outputDim, input_indices);
            int inputIdx = linearAt(inputStrides, inputDim, input_indices);
            int outputIdx = linearAt(outputStrides, outputDim, output_indices);
            output[outputIdx] = input[inputIdx];
        }
    }

    template <typename T>
    void launch_broadcastTo(const T *input, const int *inputStrides, const int intputDim,
                            const BroadcastMap *broadcastMap,
                            T *output, const int *outputStrides, const int outputDim, const int outputlen)
    {

        auto [numBlocks, blockSize] = BestDims(outputlen);

        // output
        cudaVector<int> outputStrides_d(outputStrides, outputDim, cudaMemcpyHostToDevice);

        // broadcastMap
        cudaVector<BroadcastMap> broadcastMap_d(broadcastMap, outputDim, cudaMemcpyHostToDevice);

        // input
        cudaVector<int> inputStrides_d(inputStrides, intputDim, cudaMemcpyHostToDevice);

        switch (outputDim)
        {
        case 1:
            broadcastTo_kernel<1, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 2:
            broadcastTo_kernel<2, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 3:
            broadcastTo_kernel<3, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 4:
            broadcastTo_kernel<4, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 5:
            broadcastTo_kernel<5, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 6:
            broadcastTo_kernel<6, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 7:
            broadcastTo_kernel<7, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 8:
            broadcastTo_kernel<8, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 9:
            broadcastTo_kernel<9, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 10:
            broadcastTo_kernel<10, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 11:
            broadcastTo_kernel<11, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 12:
            broadcastTo_kernel<12, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);
            break;
        default:
            throw std::runtime_error("dimension large than " + std::to_string(MAX_DIM));
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cuda error");
        }
    }
    template void launch_broadcastTo<double>(const double *input, const int *inputStrides, const int inputDim,
                                             const BroadcastMap *broadcastMap,
                                             double *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<float>(const float *input, const int *inputStrides, const int inputDim,
                                            const BroadcastMap *broadcastMap,
                                            float *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<nv_bfloat16>(const nv_bfloat16 *input, const int *inputStrides, const int inputDim,
                                                  const BroadcastMap *broadcastMap,
                                                  nv_bfloat16 *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<__half>(const __half *input, const int *inputStrides, const int inputDim,
                                             const BroadcastMap *broadcastMap,
                                             __half *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<int64_t>(const int64_t *input, const int *inputStrides, const int inputDim,
                                              const BroadcastMap *broadcastMap,
                                              int64_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<int32_t>(const int32_t *input, const int *inputStrides, const int inputDim,
                                              const BroadcastMap *broadcastMap,
                                              int32_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<int16_t>(const int16_t *input, const int *inputStrides, const int inputDim,
                                              const BroadcastMap *broadcastMap,
                                              int16_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_broadcastTo<int8_t>(const int8_t *input, const int *inputStrides, const int inputDim,
                                             const BroadcastMap *broadcastMap,
                                             int8_t *output, const int *outputStrides, const int outputDim, const int outputlen);

    // indexselect
    template <typename GatherAxisT>
    __host__ __device__ void fromIndexselectIndices(
        const int *output_indices, const int outputDim,                              // 输出张量的索引
        const GatherAxisT *index, const int *indexStrides, const int indexDim, // index是tensor
        int *index_indices,
        const int gatherAxis, // gather操作的轴
        int *input_indices, const int inputDim)
    {

        for (int i = 0; i < gatherAxis; ++i)
        {
            input_indices[i] = output_indices[i];
        }
        for (int i = gatherAxis; i < gatherAxis + indexDim; ++i)
        {
            index_indices[i - gatherAxis] = output_indices[i];
        }
        // 使用indices张量中对应位置的值来替换gatherAxis维度的索引
        int index_idx = linearAt(indexStrides, indexDim, index_indices);
        input_indices[gatherAxis] = index[index_idx];
        // for (int i = gatherAxis +indicesDim; i < outputDim; ++i)
        // {
        //     input_indices[gatherAxis+1+i] = output_indices[i];
        // }
        for (int i = 0; i < outputDim - (gatherAxis + indexDim); ++i)
        {
            input_indices[gatherAxis + 1 + i] = output_indices[gatherAxis + indexDim + i];
        }
    }

    template <int DIM, typename T, typename GatherAxisT>
    __global__ void indexselect_kernel(
        const T *input, const int *inputStrides, const int inputDim,
        const GatherAxisT *index, const int *indexStrides, const int indexDim,
        const int gatherAxis,
        T *output, const int *outputStrides, const int outputDim, const int outputlen)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < outputlen; thread_id += grid_stride)
        {
            // 输出索引
            int output_indices[DIM];
            linearTo(outputStrides, outputDim, output_indices, thread_id);

            // 输入索引
            int index_indices[DIM];
            int input_indices[DIM];
            fromIndexselectIndices(output_indices, outputDim,
                                   index, indexStrides, indexDim,
                                   index_indices,
                                   gatherAxis,
                                   input_indices, inputDim);
            int inputIdx = linearAt(inputStrides, inputDim, input_indices);
            int outputIdx = linearAt(outputStrides, outputDim, output_indices);
            output[outputIdx] = input[inputIdx];
        }
    }

    template <typename T, typename GatherAxisT>
    void launch_indexselect(
        const T *input, const int *inputStrides, const int inputDim,
        const GatherAxisT *index, const int *indexStrides, const int indexDim,
        const int gatherAxis,
        T *output, const int *outputStrides, const int outputDim, const int outputlen)
    {

        auto [numBlocks, blockSize] = BestDims(outputlen);

        // indices
        cudaVector<int> indexStrides_d(indexStrides, indexDim, cudaMemcpyHostToDevice);

        // input
        cudaVector<int> inputStrides_d(inputStrides, inputDim, cudaMemcpyHostToDevice);

        // output
        cudaVector<int> outputStrides_d(outputStrides, outputDim, cudaMemcpyHostToDevice);

        // TODO 这里可能会导致寄存器浪费，但是，搞太多模板T，模板实例化不好搞
        int dim = std::max(inputDim, indexDim);
        dim = std::max(dim, outputDim);
        switch (dim)
        {
        case 1:
            indexselect_kernel<1, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 2:
            indexselect_kernel<2, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 3:
            indexselect_kernel<3, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 4:
            indexselect_kernel<4, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 5:
            indexselect_kernel<5, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 6:
            indexselect_kernel<6, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 7:
            indexselect_kernel<7, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 8:
            indexselect_kernel<8, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 9:
            indexselect_kernel<9, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 10:
            indexselect_kernel<10, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 11:
            indexselect_kernel<11, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        case 12:
            indexselect_kernel<12, T, GatherAxisT><<<numBlocks, blockSize>>>(input, inputStrides_d.data, inputDim, index, indexStrides_d.data, indexDim, gatherAxis, output, outputStrides_d.data, outputDim, outputlen);
            break;
        default:
            throw std::runtime_error("dimension large than " + std::to_string(MAX_DIM));
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cuda error");
        }
    }
    template void launch_indexselect<double, int64_t>(const double *input, const int *inputStrides, const int inputDim,
                                                      const int64_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      double *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<float, int64_t>(const float *input, const int *inputStrides, const int inputDim,
                                                     const int64_t *index, const int *indexStrides, const int indexDim,
                                                     const int gatherAxis,
                                                     float *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<nv_bfloat16, int64_t>(const nv_bfloat16 *input, const int *inputStrides, const int inputDim,
                                                           const int64_t *index, const int *indexStrides, const int indexDim,
                                                           const int gatherAxis,
                                                           nv_bfloat16 *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<__half, int64_t>(const __half *input, const int *inputStrides, const int inputDim,
                                                      const int64_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      __half *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int64_t, int64_t>(const int64_t *input, const int *inputStrides, const int inputDim,
                                                       const int64_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int64_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int32_t, int64_t>(const int32_t *input, const int *inputStrides, const int inputDim,
                                                       const int64_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int32_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int16_t, int64_t>(const int16_t *input, const int *inputStrides, const int inputDim,
                                                       const int64_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int16_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int8_t, int64_t>(const int8_t *input, const int *inputStrides, const int inputDim,
                                                      const int64_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      int8_t *output, const int *outputStrides, const int outputDim, const int outputlen);

    template void launch_indexselect<double, int32_t>(const double *input, const int *inputStrides, const int inputDim,
                                                        const int32_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      double *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<float, int32_t>(const float *input, const int *inputStrides, const int inputDim,
                                                     const int32_t *index, const int *indexStrides, const int indexDim,
                                                     const int gatherAxis,
                                                     float *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<nv_bfloat16, int32_t>(const nv_bfloat16 *input, const int *inputStrides, const int inputDim,
                                                           const int32_t *index, const int *indexStrides, const int indexDim,
                                                           const int gatherAxis,
                                                           nv_bfloat16 *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<__half, int32_t>(const __half *input, const int *inputStrides, const int inputDim,
                                                      const int32_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      __half *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int64_t, int32_t>(const int64_t *input, const int *inputStrides, const int inputDim,
                                                       const int32_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int64_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int32_t, int32_t>(const int32_t *input, const int *inputStrides, const int inputDim,
                                                       const int32_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int32_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int16_t, int32_t>(const int16_t *input, const int *inputStrides, const int inputDim,
                                                       const int32_t *index, const int *indexStrides, const int indexDim,
                                                       const int gatherAxis,
                                                       int16_t *output, const int *outputStrides, const int outputDim, const int outputlen);
    template void launch_indexselect<int8_t, int32_t>(const int8_t *input, const int *inputStrides, const int inputDim,
                                                      const int32_t *index, const int *indexStrides, const int indexDim,
                                                      const int gatherAxis,
                                                      int8_t *output, const int *outputStrides, const int outputDim, const int outputlen);
}

#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP