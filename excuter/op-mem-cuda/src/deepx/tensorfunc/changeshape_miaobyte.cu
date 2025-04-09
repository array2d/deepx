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
    //  DIM=2^n
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

    inline int nextPowerOf2(int n)
    {
        if (n <= 0)
            return 1;
        if ((n & (n - 1)) == 0)
            return n; // 如果n已经是2的幂

        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

    template <typename T>
    void launch_transpose(const int numBlocks, const int blockSize,
                          const T *input,
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

        int powDim = nextPowerOf2(dim);

        // 根据计算出的2的幂次选择对应的模板实例
        switch (powDim)
        {
        case 1:
            transpose_kernel<1, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 2:
            transpose_kernel<2, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 4:
            transpose_kernel<4, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 8:
            transpose_kernel<8, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 16:
            transpose_kernel<16, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 32:
            transpose_kernel<32, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 64:
            transpose_kernel<64, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        case 128:
            transpose_kernel<128, T><<<numBlocks, blockSize>>>(input, strides_d.data, output, newStrides_d.data, dim, len, dimOrder_d.data);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
        }
    }

    template void launch_transpose<double>(const int numBlocks, const int blockSize, const double *input, const int *inputStrides, double *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<float>(const int numBlocks, const int blockSize, const float *input, const int *inputStrides, float *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<nv_bfloat16>(const int numBlocks, const int blockSize, const nv_bfloat16 *input, const int *inputStrides, nv_bfloat16 *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<__half>(const int numBlocks, const int blockSize, const __half *input, const int *inputStrides, __half *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int64_t>(const int numBlocks, const int blockSize, const int64_t *input, const int *inputStrides, int64_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int32_t>(const int numBlocks, const int blockSize, const int32_t *input, const int *inputStrides, int32_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int16_t>(const int numBlocks, const int blockSize, const int16_t *input, const int *inputStrides, int16_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);
    template void launch_transpose<int8_t>(const int numBlocks, const int blockSize, const int8_t *input, const int *inputStrides, int8_t *output, const int *outputStrides, const int dim, const int len, const int *dimOrder);

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

        int powDim = nextPowerOf2(dim);

        // 根据计算出的2的幂次选择对应的模板实例
        switch (powDim)
        {
        case 1:
            concat_kernel<1, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 2:
            concat_kernel<2, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 4:
            concat_kernel<4, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 8:
            concat_kernel<8, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 16:
            concat_kernel<16, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 32:
            concat_kernel<32, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 64:
            concat_kernel<64, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        case 128:
            concat_kernel<128, T><<<numBlocks, blockSize>>>(tensorsDataList.data, inputStrides_d.data, outputData, outputStrides_d.data, dim, outputLen, axis, numTensors, shapeAtAxis_d.data);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
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
                            T *output, const int *outputStrides, const int outputDim, const int outputlen){

        auto [numBlocks, blockSize] = BestDims(outputlen);

        // output
        cudaVector<int> outputStrides_d(outputStrides, outputDim, cudaMemcpyHostToDevice);

        // broadcastMap
        cudaVector<BroadcastMap> broadcastMap_d(broadcastMap, outputDim, cudaMemcpyHostToDevice);

        // input
        cudaVector<int> inputStrides_d(inputStrides, intputDim, cudaMemcpyHostToDevice);

     
        int powDim = nextPowerOf2(outputDim);   
        // 根据计算出的2的幂次选择对应的模板实例
        switch (powDim)
        {
        case 1:
            broadcastTo_kernel<1, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);    
            break;
        case 2:
            broadcastTo_kernel<2, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);    
            break;
        case 4:
            broadcastTo_kernel<4, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);    
            break;
        case 8:
            broadcastTo_kernel<8, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);    
            break;
        case 16:
            broadcastTo_kernel<16, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);       
            break;
        case 32:
            broadcastTo_kernel<32, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);       
            break;
        case 64:
            broadcastTo_kernel<64, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);       
            break;
        case 128:
            broadcastTo_kernel<128, T><<<numBlocks, blockSize>>>(input, inputStrides_d.data, intputDim, broadcastMap_d.data, output, outputStrides_d.data, outputDim, outputlen);       
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
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
                                             __half *output,     const int *outputStrides, const int outputDim, const int outputlen);
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
}
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP