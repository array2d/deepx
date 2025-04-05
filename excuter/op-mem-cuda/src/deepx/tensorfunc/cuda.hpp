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

    inline std::pair<int, int> BestDims(int total_elements)
    {
        // 默认块大小
        int optimal_block_size = 256; // 一般256或512是较好的选择
        // 计算设备属性以确定最佳配置
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        // 根据SM数量和每个SM的最大线程数决定块数
        int sm_count = props.multiProcessorCount;
        int optimal_blocks = sm_count * 8; // 每个SM分配多个块以增加并行度

        // 确保至少启动足够的线程来处理所有数据
        int min_blocks = (total_elements + optimal_block_size - 1) / optimal_block_size;
        int actual_blocks = std::min(optimal_blocks, min_blocks);

        return {actual_blocks, optimal_block_size};
    };
}

#endif
