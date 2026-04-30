#ifndef DEEPX_THREAD_PARALLEL_HPP
#define DEEPX_THREAD_PARALLEL_HPP

#include <vector>
#include <thread>
#include <stdexcept>
#include "deepx/shape.hpp"

namespace deepx::thread
{

struct ThreadLocalVectors
{
private:
    std::vector<std::vector<int>> vectors;

public:
    explicit ThreadLocalVectors(const std::vector<int> &sizes)
    {
        vectors.resize(sizes.size());
        for (size_t i = 0; i < sizes.size(); ++i)
            vectors[i].resize(sizes[i], 0);
    }
    std::vector<int> &get(size_t index) { return vectors[index]; }
    std::vector<std::vector<int>> &getAll() { return vectors; }
};

inline int checkdim(int dimCount, int dim)
{
    if (dimCount < 0) dimCount += dim;
    if (dimCount > dim)
        throw std::invalid_argument("dimCount exceeds the number of dimensions");
    return dimCount;
}

inline int checkTotalSize(int dimCount, const std::vector<int> &shape)
{
    int totalSize = 1;
    for (int i = 0; i < dimCount; ++i) totalSize *= shape[i];
    return totalSize;
}

inline int checkStride(int dimCount, const std::vector<int> &shape)
{
    int stride = 1;
    for (int i = dimCount; i < (int)shape.size(); ++i) stride *= shape[i];
    return stride;
}

// ── rangeParallel (indices only) ──
template <typename Func>
void rangeParallel(const Shape &s, int dimCount, Func &&func)
{
    dimCount = checkdim(dimCount, s.dim());
    int totalSize = checkTotalSize(dimCount, s.shape);
#pragma omp parallel
    {
        std::vector<int> indices(dimCount, 0);
#pragma omp for
        for (int idx = 0; idx < totalSize; idx++)
        {
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % s.shape[dim];
                idx_ /= s.shape[dim];
            }
            func(indices);
        }
    }
}

// ── rangeParallel (linear + indices) ──
template <typename Func>
void rangeParallelLinear(const Shape &s, int dimCount, Func &&func)
{
    dimCount = checkdim(dimCount, s.dim());
    int totalSize = checkTotalSize(dimCount, s.shape);
    int stride = checkStride(dimCount, s.shape);
#pragma omp parallel
    {
        std::vector<int> indices(dimCount, 0);
#pragma omp for
        for (int idx = 0; idx < totalSize; idx++)
        {
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % s.shape[dim];
                idx_ /= s.shape[dim];
            }
            func(idx * stride, indices);
        }
    }
}

// ── rangeElementwiseParallel ──
template <typename Func>
void rangeElementwiseParallel(const Shape &s, Func &&func)
{
    int num_threads = std::thread::hardware_concurrency();
    int alignblock = s.size / num_threads;
    const int minblock = 256;
    if (alignblock < minblock)
    {
        alignblock = minblock;
        num_threads = s.size / alignblock;
    }
#pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < s.size; idx += alignblock)
    {
        int end = idx + alignblock;
        if (end > s.size) end = s.size;
        func(idx, end);
    }
}

// ── rangeParallel + ThreadLocalVectors (indices) ──
template <typename Func>
void rangeParallel(const Shape &s, int dimCount, Func &&func,
                   const std::vector<int> &tlv_sizes)
{
    dimCount = checkdim(dimCount, s.dim());
    int totalSize = checkTotalSize(dimCount, s.shape);
#pragma omp parallel
    {
        std::vector<int> indices(dimCount, 0);
        ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
        for (int idx = 0; idx < totalSize; idx++)
        {
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % s.shape[dim];
                idx_ /= s.shape[dim];
            }
            func(indices, tlv);
        }
    }
}

// ── rangeParallel + ThreadLocalVectors (linear only) ──
template <typename Func>
void rangeParallelLinear(const Shape &s, int dimCount, Func &&func,
                         const std::vector<int> &tlv_sizes)
{
    dimCount = checkdim(dimCount, s.dim());
    int stride = checkStride(dimCount, s.shape);
    int total = s.size / stride;
#pragma omp parallel
    {
        ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
        for (int idx = 0; idx < total; idx++)
            func(idx * stride, tlv);
    }
}

// ── rangeParallel + ThreadLocalVectors (linear + indices) ──
template <typename Func>
void rangeParallelMixed(const Shape &s, int dimCount, Func &&func,
                        const std::vector<int> &tlv_sizes)
{
    dimCount = checkdim(dimCount, s.dim());
    int totalSize = checkTotalSize(dimCount, s.shape);
    int stride = checkStride(dimCount, s.shape);
#pragma omp parallel
    {
        std::vector<int> indices(dimCount, 0);
        ThreadLocalVectors tlv(tlv_sizes);
#pragma omp for
        for (int idx = 0; idx < totalSize; idx++)
        {
            int idx_ = idx;
            for (int dim = dimCount - 1; dim >= 0; --dim)
            {
                indices[dim] = idx_ % s.shape[dim];
                idx_ /= s.shape[dim];
            }
            func(idx * stride, indices, tlv);
        }
    }
}

} // namespace deepx::thread
#endif
