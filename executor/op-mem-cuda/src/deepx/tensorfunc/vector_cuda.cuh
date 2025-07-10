#ifndef DEEPX_TENSORFUNC_VECTOR_CUDA_CUH
#define DEEPX_TENSORFUNC_VECTOR_CUDA_CUH

namespace deepx::tensorfunc
{
    //TODO 待验证
    template <typename T>
    __device__ void GridStrideLoopCopy(const T* src, T* dst, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        
        for (int i = idx; i < size; i += stride) {
            dst[i] = src[i];
        }
    }
     //TODO 待验证
    // 全局复制函数，可从主机调用
    template <typename T>
    __global__ void GridStrideLoopCopyKernel(const T* src, T* dst, int size) {
        GridStrideLoopCopy(src, dst, size);
    }

    //cudaVector
    template <typename T>
    struct cudaVector
    {
        T *data;
        int size;
        __device__ __host__ cudaVector(int size) : size(size)
        {
            cudaMalloc(&data, size * sizeof(T));
        }
        __host__ cudaVector(const T *src, int size, cudaMemcpyKind kind = cudaMemcpyHostToDevice) : size(size)
        {
            cudaMalloc(&data, size * sizeof(T));
            cudaMemcpy(data, src, size * sizeof(T), kind);
        }
        __host__ cudaVector(const cudaVector &other) : size(other.size)
        {
            cudaMalloc(&data, size * sizeof(T));
            cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        __device__ __host__ cudaVector(cudaVector &&other) noexcept : data(other.data), size(other.size)
        {
            other.data = nullptr;
            other.size = 0;
        }
        __device__ __host__ cudaVector &operator=(const cudaVector &other)
        {
            if (this != &other)
            {
                cudaFree(data);
                data = other.data;
                size = other.size;
            }
            return *this;
        }
        __device__ __host__ cudaVector &operator=(cudaVector &&other) noexcept
        {
            if (this != &other)
            {
                cudaFree(data);
                data = other.data;
                size = other.size;
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }
        __device__ __host__ ~cudaVector()
        {
            cudaFree(data);
        }
        __device__ __host__ void copyFromHost(const T *hostData, int size,int offset=0)
        {
            cudaMemcpy(data+offset, hostData, size * sizeof(T), cudaMemcpyHostToDevice);
        }
        __device__ __host__ void copyToHost(T *hostData, int size,int offset=0)
        {
            cudaMemcpy(hostData, data+offset, size * sizeof(T), cudaMemcpyDeviceToHost);
        }
        __device__ __host__ void copyFromDevice(const T *deviceData, int size,int offset=0)
        {
            for (int i = 0; i < size; i++)
            {
                data[offset+i] = deviceData[i];
            }
        }
        __device__ __host__ T &operator[](int idx)
        {
            return data[idx];
        }
        __device__ __host__ const T &operator[](int idx) const
        {
            return data[idx];
        }
 
    };
}

#endif // DEEPX_TENSORFUNC_VECTOR_CUDA_CUH
