#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CU
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/reduce_miaobyte.cuh"
#include "deepx/tensorfunc/tensor_cuda.cuh"
#include "deepx/tensorfunc/vector_cuda.cuh"

#include "deepx/tensorfunc/cuda_atomic.cuh"
#include "deepx/tensorfunc/cuda_math.cuh"
namespace deepx::tensorfunc
{

 
    // sum
    //DIM是希望申请寄存器中存放索引数组的长度
    template <int DIM, typename T>
    __global__ void sum_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                               const int *reduced_dims, const bool keepdims,
                               T *result_data, const int *result_strides, const int result_dim)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < tensor_len; thread_id += grid_stride)
        {
            int input_indices[DIM];
            linearTo(tensor_strides, tensor_dim, input_indices, thread_id);
            int output_indices[DIM];
            for (size_t i = 0, j = 0; i < tensor_dim; ++i)
            {
                if (reduced_dims[i] == 0)
                {
                    output_indices[j++] = input_indices[i];
                }
                else if (keepdims && (reduced_dims[i] == 1))
                {
                    output_indices[j++] = 0;
                }
            }
            int outputIdx = linearAt(result_strides, result_dim, output_indices);
            int inputIdx = linearAt(tensor_strides, tensor_dim, input_indices);
            deepx_atomicAdd(result_data + outputIdx, tensor_data[inputIdx]);
        }
    }

    template <typename T>
    __host__ void launch_sum(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                             const int *reduced_dims, const bool keepdims,
                             T *result_data, const int *result_strides, const int result_dim)
    {
        auto [numBlocks, blockSize] = BestDims(tensor_len);
        // int shared_mem_size = blockSize * sizeof(T) + sizeof(int) * tensor_dim;
        cudaVector<int> tensor_strides_d(tensor_strides, tensor_dim, cudaMemcpyHostToDevice);
        cudaVector<int> result_strides_d(result_strides, result_dim, cudaMemcpyHostToDevice);
        cudaVector<int> reduced_dims_d(reduced_dims, tensor_dim, cudaMemcpyHostToDevice);

        int powDim = nextPowerOf2(tensor_dim);
        switch (powDim)
        {
        case 1:
            sum_kernel<1, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 2:
            sum_kernel<2, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 4:
            sum_kernel<4, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 8:
            sum_kernel<8, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 16:
            sum_kernel<16, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 32:
            sum_kernel<32, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);    
            break;
        case 64:
            sum_kernel<64, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 128:
            sum_kernel<128, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
        }
    }

    template void launch_sum<double>(const double *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                     const int *reduced_dims, const bool keepdims,
                                     double *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<float>(const float *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                    const int *reduced_dims, const bool keepdims,
                                    float *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<nv_bfloat16>(const nv_bfloat16 *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                          const int *reduced_dims, const bool keepdims,
                                          nv_bfloat16 *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<__half>(const __half *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                     const int *reduced_dims, const bool keepdims,
                                     __half *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<int64_t>(const int64_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      int64_t *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<int32_t>(const int32_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      int32_t *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<int16_t>(const int16_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      int16_t *result_data, const int *result_strides, const int result_dim);
    template void launch_sum<int8_t>(const int8_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                     const int *reduced_dims, const bool keepdims,
                                     int8_t *result_data, const int *result_strides, const int result_dim);

    // prod
    template <int DIM, typename T>
    __global__ void prod_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                const int *reduced_dims, const bool keepdims,
                                T *result_data, const int *result_strides, const int result_dim)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < tensor_len; thread_id += grid_stride)
        {
            int input_indices[DIM];
            linearTo(tensor_strides, tensor_dim, input_indices, thread_id);
            int output_indices[DIM];
            for (size_t i = 0, j = 0; i < tensor_dim; ++i)
            {
                if (reduced_dims[i] == 0)
                {
                    output_indices[j++] = input_indices[i];
                }
                else if (keepdims && (reduced_dims[i] == 1))
                {
                    output_indices[j++] = 0;
                }
            }
            int outputIdx = linearAt(result_strides, result_dim, output_indices);
            int inputIdx = linearAt(tensor_strides, tensor_dim, input_indices);
            deepx_atomicMul(result_data + outputIdx, tensor_data[inputIdx]);
        }
    }

    template <typename T>
    void launch_prod(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                     const int *reduced_dims, const bool keepdims,
                     T *result_data, const int *result_strides, const int result_dim)
    {
        auto [numBlocks, blockSize] = BestDims(tensor_len);
        cudaVector<int> tensor_strides_d(tensor_strides, tensor_dim, cudaMemcpyHostToDevice);
        cudaVector<int> result_strides_d(result_strides, result_dim, cudaMemcpyHostToDevice);
        cudaVector<int> reduced_dims_d(reduced_dims, tensor_dim, cudaMemcpyHostToDevice);

        int powDim = nextPowerOf2(tensor_dim);
        switch (powDim)
        {
        case 1:
            prod_kernel<1, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 2:
            prod_kernel<2, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 4:
            prod_kernel<4, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 8:
            prod_kernel<8, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 16:
            prod_kernel<16, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 32:
            prod_kernel<32, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 64:
            prod_kernel<64, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 128:
            prod_kernel<128, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
        }
    }

    template void launch_prod<double>(const double *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      double *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<float>(const float *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                     const int *reduced_dims, const bool keepdims,
                                     float *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<nv_bfloat16>(const nv_bfloat16 *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           nv_bfloat16 *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<__half>(const __half *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      __half *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<int64_t>(const int64_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                       const int *reduced_dims, const bool keepdims,
                                       int64_t *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<int32_t>(const int32_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                       const int *reduced_dims, const bool keepdims,
                                       int32_t *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<int16_t>(const int16_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                       const int *reduced_dims, const bool keepdims,
                                       int16_t *result_data, const int *result_strides, const int result_dim);
    template void launch_prod<int8_t>(const int8_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      int8_t *result_data, const int *result_strides, const int result_dim);

    // max
    template <int DIM, typename T>
    __global__ void max_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                               const int *reduced_dims, const bool keepdims,
                               T *result_data, const int *result_strides, const int result_dim)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < tensor_len; thread_id += grid_stride)
        {
            int input_indices[DIM];
            linearTo(tensor_strides, tensor_dim, input_indices, thread_id);
            int output_indices[DIM];
            for (size_t i = 0, j = 0; i < tensor_dim; ++i)
            {
                if (reduced_dims[i] == 0)
                {
                    output_indices[j++] = input_indices[i];
                }
                else if (keepdims && (reduced_dims[i] == 1))
                {
                    output_indices[j++] = 0;
                }
            }
            int outputIdx = linearAt(result_strides, result_dim, output_indices);
            int inputIdx = linearAt(tensor_strides, tensor_dim, input_indices);
            deepx_max(result_data + outputIdx, tensor_data + inputIdx, result_data + outputIdx);
        }
    }

    template <typename T>
    void launch_reducemax(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                          const int *reduced_dims, const bool keepdims,
                          T *result_data, const int *result_strides, const int result_dim)
    {
        auto [numBlocks, blockSize] = BestDims(tensor_len);
        cudaVector<int> tensor_strides_d(tensor_strides, tensor_dim, cudaMemcpyHostToDevice);
        cudaVector<int> result_strides_d(result_strides, result_dim, cudaMemcpyHostToDevice);
        cudaVector<int> reduced_dims_d(reduced_dims, tensor_dim, cudaMemcpyHostToDevice);

        int powDim = nextPowerOf2(tensor_dim);
        switch (powDim)
        {
        case 1:
            max_kernel<1, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 2:
            max_kernel<2, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 4:
            max_kernel<4, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 8:
            max_kernel<8, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 16:
            max_kernel<16, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 32:
            max_kernel<32, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 64:
            max_kernel<64, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 128:
            max_kernel<128, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
        }
    };

    template void launch_reducemax<double>(const double *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           double *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<float>(const float *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                          const int *reduced_dims, const bool keepdims,
                                          float *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<nv_bfloat16>(const nv_bfloat16 *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                                const int *reduced_dims, const bool keepdims,
                                                nv_bfloat16 *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<__half>(const __half *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           __half *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<int64_t>(const int64_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int64_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<int32_t>(const int32_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int32_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<int16_t>(const int16_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int16_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemax<int8_t>(const int8_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           int8_t *result_data, const int *result_strides, const int result_dim);

    // min
    template <int DIM, typename T>
    __global__ void min_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                               const int *reduced_dims, const bool keepdims,
                               T *result_data, const int *result_strides, const int result_dim)
    {
        const int grid_stride = gridDim.x * blockDim.x;
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (; thread_id < tensor_len; thread_id += grid_stride)
        {
            int input_indices[DIM];
            linearTo(tensor_strides, tensor_dim, input_indices, thread_id);
            int output_indices[DIM];
            for (size_t i = 0, j = 0; i < tensor_dim; ++i)
            {
                if (reduced_dims[i] == 0)
                {
                    output_indices[j++] = input_indices[i];
                }
                else if (keepdims && (reduced_dims[i] == 1))
                {
                    output_indices[j++] = 0;
                }
            }
            int outputIdx = linearAt(result_strides, result_dim, output_indices);
            int inputIdx = linearAt(tensor_strides, tensor_dim, input_indices);
            deepx_min(result_data + outputIdx, tensor_data + inputIdx, result_data + outputIdx);
        }
    }

    template <typename T>
    void launch_reducemin(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                          const int *reduced_dims, const bool keepdims,
                          T *result_data, const int *result_strides, const int result_dim)
    {
        auto [numBlocks, blockSize] = BestDims(tensor_len);
        cudaVector<int> tensor_strides_d(tensor_strides, tensor_dim, cudaMemcpyHostToDevice);
        cudaVector<int> result_strides_d(result_strides, result_dim, cudaMemcpyHostToDevice);
        cudaVector<int> reduced_dims_d(reduced_dims, tensor_dim, cudaMemcpyHostToDevice);

        int powDim = nextPowerOf2(tensor_dim);
        switch (powDim)
        {
        case 1:
            min_kernel<1, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 2:
            min_kernel<2, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 4:
            min_kernel<4, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 8:
            min_kernel<8, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 16:
            min_kernel<16, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 32:
            min_kernel<32, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 64:
            min_kernel<64, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        case 128:
            min_kernel<128, T><<<numBlocks, blockSize>>>(tensor_data, tensor_strides_d.data, tensor_dim, tensor_len, reduced_dims_d.data, keepdims, result_data, result_strides_d.data, result_dim);
            break;
        default:
            throw std::runtime_error("dim too large, max support 128");
        }
    }

    template void launch_reducemin<double>(const double *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           double *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<float>(const float *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                          const int *reduced_dims, const bool keepdims,
                                          float *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<nv_bfloat16>(const nv_bfloat16 *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                                const int *reduced_dims, const bool keepdims,
                                                nv_bfloat16 *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<__half>(const __half *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           __half *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<int64_t>(const int64_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int64_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<int32_t>(const int32_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int32_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<int16_t>(const int16_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                            const int *reduced_dims, const bool keepdims,
                                            int16_t *result_data, const int *result_strides, const int result_dim);
    template void launch_reducemin<int8_t>(const int8_t *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                           const int *reduced_dims, const bool keepdims,
                                           int8_t *result_data, const int *result_strides, const int result_dim);
}

#endif // DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CU
