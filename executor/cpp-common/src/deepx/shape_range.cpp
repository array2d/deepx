#include <iostream>
#include <vector>
#include <functional>
#include <any>
#include <thread>

#include <omp.h>
#include "deepx/shape.hpp"
namespace deepx
{
    static int checkdim(int dimCount, int dim)
    {
        if (dimCount < 0)
        {
            dimCount += dim;
        }
        if (dimCount > dim)
        {
            throw std::invalid_argument("dimCount exceeds the number of dimensions in the Tensor.");
        }
        return dimCount;
    }
    static int checkTotalSize(int dimCount, std::vector<int> shape)
    {
        int totalSize = 1;
        for (int i = 0; i < dimCount; ++i)
        {
            totalSize *= shape[i];
        }
        return totalSize;
    }

    static int checkStride(int dimCount, std::vector<int> shape)
    {
        int stride = 1;

        // 计算总的循环次数
        for (int i = dimCount; i < shape.size(); ++i)
        {
            stride *= shape[i];
        }
        return stride;
    }
    void Shape::range(int dimCount, std::function<void(const std::vector<int> &indices)> func) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);

        std::vector<int> indices(dimCount, 0);

        for (int idx = 0; idx < totalSize; idx++)
        {
            // 反算出 indices 数组
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                idx_ /= shape[dim];               // 更新 idx
            }
            func(indices); // 调用传入的函数
        }
    }
    void Shape::range(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);

        int stride = checkStride(dimCount, shape);

        std::vector<int> indices(dimCount, 0);

        for (int idx = 0; idx < totalSize; idx++)
        {
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                idx_ /= shape[dim];               // 更新 idx
            }
            func(idx * stride, indices);
        }
    }

    void Shape::range(int dimCount, std::function<void(const int idx_linear)> func) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);
        int stride = checkStride(dimCount, shape);
        for (int idx = 0; idx < totalSize; idx++)
        {
            func(idx * stride);
        }
    }

    void Shape::rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices)> func) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);

#pragma omp parallel
        {
            std::vector<int> indices(dimCount, 0);
#pragma omp for
            for (int idx = 0; idx < totalSize; idx++)
            {
                // 反算出 indices 数组
                int idx_ = idx;
                for (int dim = dimCount - 1; dim >= 0; --dim)
                {
                    indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                    idx_ /= shape[dim];               // 更新 idx
                }
                func(indices); // 调用传入的函数
            }
        }
    }
    void Shape::rangeElementwiseParallel(std::function<void(const int idx_linear,const int idx_linear_end)> func) const
    {   
        int num_threads =  std::thread::hardware_concurrency();
        int alignblock=size/num_threads;
        const int minblock=256;
        if (alignblock<minblock)
        {
            alignblock=minblock;
            num_threads=size/alignblock;
        }
        #pragma omp parallel for num_threads(num_threads)
        for (int idx = 0; idx < size; idx+=alignblock)
        {
            int end = idx + alignblock;
            if (end > size) {
                end = size;
            }
            func(idx,end);
        }
    }

    void Shape::rangeParallel(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);
        int stride = checkStride(dimCount, shape);

#pragma omp parallel
        {
            std::vector<int> indices(dimCount, 0);
#pragma omp for
            for (int idx = 0; idx < totalSize; idx++)
            {
                // printf("线程 %d 处理索引 %d\n", omp_get_thread_num(), idx);
                int idx_ = idx;
                for (int dim = dimCount - 1; dim >= 0; --dim)
                {
                    indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                    idx_ /= shape[dim];               // 更新 idx
                }
                func(idx * stride, indices);
            }
        }
    }

    void Shape::rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);

#pragma omp parallel
        {
            std::vector<int> indices(dimCount, 0);
            ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
            for (int idx = 0; idx < totalSize; idx++)
            {
                // 反算出 indices 数组
                int idx_ = idx;
                for (int dim = dimCount - 1; dim >= 0; --dim)
                {
                    indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                    idx_ /= shape[dim];               // 更新 idx
                }
                func(indices, tlv); // 调用传入的函数
            }
        }
    }
    void Shape::rangeParallel(int dimCount, std::function<void(const int idx_linear, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const
    {
        dimCount = checkdim(dimCount, dim());
        int stride = checkStride(dimCount, shape);

        // 计算总循环次数
        int total = size / stride;

#pragma omp parallel
        {
            ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
            for (int idx = 0; idx < total; idx++)
            {
                func(idx * stride, tlv);
            }
        }
    }

    void Shape::rangeParallel(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const
    {
        dimCount = checkdim(dimCount, dim());
        int totalSize = checkTotalSize(dimCount, shape);
        int stride = checkStride(dimCount, shape);

#pragma omp parallel
        {
            std::vector<int> indices(dimCount, 0);
            ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
            for (int idx = 0; idx < totalSize; idx++)
            {
                // printf("线程 %d 处理索引 %d\n", omp_get_thread_num(), idx);
                int idx_ = idx;
                for (int dim = dimCount - 1; dim >= 0; --dim)
                {
                    indices[dim] = idx_ % shape[dim]; // 计算当前维度的索引
                    idx_ /= shape[dim];               // 更新 idx
                }
                func(idx * stride, indices, tlv);
            }
        }
    }
}