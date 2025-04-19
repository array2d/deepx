#ifndef DEEPX_TENSORFUNC_CUDA_HPP
#define DEEPX_TENSORFUNC_CUDA_HPP

#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>

namespace deepx::tensorfunc
{
    class CublasHandle
    {
    public:
        CublasHandle()
        {
            if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
        }
        ~CublasHandle()
        {
            if (handle_)
                cublasDestroy(handle_);
        }

        cublasHandle_t get() { return handle_; }

    private:
        cublasHandle_t handle_;
    };
    //TODO 
    inline int deviceblocksize(){
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        return props.maxThreadsPerBlock;
    }
    inline int deviceblock()
    {
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        // 根据SM数量计算建议的块数上限
        int sm_count = props.multiProcessorCount;
        int optimal_blocks = sm_count * 8; // 每个SM分配多个块以增加并行度
        return optimal_blocks;
    }

    // 计算最佳的块大小和块数
    inline std::pair<int, int> BestDims(int total_elements)
    {
        // 默认块大小
        int blocksize = total_elements > 256 ? 256 : total_elements;
        int blocks = (total_elements + blocksize - 1) / blocksize; // 向上取整除法
        int optimal_blocks = deviceblock();
        blocks = std::min(blocks, optimal_blocks);
        return {blocks, blocksize};
    };
}

#endif
