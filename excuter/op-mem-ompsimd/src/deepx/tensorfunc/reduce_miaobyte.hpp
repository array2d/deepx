#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP

#include <vector>
#include <stdexcept>
#include <hwy/highway.h>

#include "deepx/tensorfunc/highway.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"

namespace deepx::tensorfunc
{
    using namespace hwy::HWY_NAMESPACE;

    // sum author=miaobyte
    template <typename T>
    struct sumDispatcher<miaobyte, T>
    {
        static void sum(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            constant<miaobyte, T>(result, T(0));
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            const int minshape_1 = Lanes(ScalableTag<T>());
            if (checkeddims.rbegin()[0] == tensor.shape.dim() - 1 || tensor.shape.dim() > reduced_dims.size() || tensor.shape[-1] >= minshape_1)
            {
                tensor.shape.rangeParallel(tensor.shape.dim(), [&tensor, &result, &reduced_dims, keepdims](const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                    // 计算输出索引
                    for (size_t i = 0, j = 0; i < tensor.shape.dim(); ++i)
                    {
                        if (reduced_dims[i] == 0)
                        {
                            tlv.get(0)[j++] = indices[i];
                        }else if (keepdims && (reduced_dims[i] == 1)) {
                            tlv.get(0)[j++] = 0;
                        }
                    }
                    int outputIdx = result.shape.linearat(tlv.get(0));
#pragma omp atomic
                    result.data[outputIdx] += tensor.data[idx_linear]; }, {result.shape.dim()});
            }
            else
            {
                // 如果数据连续（对齐），则可以simd
                tensor.shape.rangeParallel(tensor.shape.dim() - 1, [&tensor, &result, &reduced_dims, keepdims](const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                    // 计算输出索引
                    for (size_t i = 0, j = 0; i < tensor.shape.dim(); ++i)
                    {
                        if (reduced_dims[i] == 0)
                        {
                            tlv.get(0)[j++] = indices[i];
                        }else if (keepdims && (reduced_dims[i] == 1)) {
                            tlv.get(0)[j++] = 0;
                        }
                    }
                    int outputIdx = result.shape.linearat(tlv.get(0));
                    int shape_last = tensor.shape[-1];
                    const ScalableTag<T> tag;
                    const size_t lanes = Lanes(tag);
                    size_t j = 0;
                    T sum = 0;
                    // 前部分：处理到对齐
                    while (j < shape_last && !IsAligned(tag, tensor.data + idx_linear + j))
                    {
                        sum += tensor.data[idx_linear + j];
                        ++j;    
                    }
                    // 中间部分：SIMD
                    size_t aligned_end = shape_last - (shape_last % lanes);
                    auto sum_vec = Zero(tag);
                    for (; j + lanes <= aligned_end; j += lanes)
                    {
                        auto vec = Load(tag, tensor.data + idx_linear + j);
                        sum_vec = Add(sum_vec, vec);
                    }
                    // 将向量累加结果写回
                    sum += ReduceSum(tag, sum_vec);
                    // 尾部分：处理剩余
                    for (; j < shape_last; ++j)
                    {
                        sum += tensor.data[idx_linear + j];
                    }
#pragma omp atomic
                    result.data[outputIdx] += sum; },
                     {result.shape.dim()});
            }
        }
    };

    // prod author=miaobyte
    template <typename T>
    struct prodDispatcher<miaobyte, T>
    {
        static void prod(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            const int minshape_1 = Lanes(ScalableTag<T>());
            // 如果dims的最后一个元素是tensor.shape.dim-1，则说明reduceprod的数据不连续（不对齐），无法simd（需要不停跳跃）
            constant<miaobyte, T>(result, T(1));
            if (reduced_dims.rbegin()[0] == tensor.shape.dim() - 1 || tensor.shape.dim() > reduced_dims.size() || tensor.shape[-1] >= minshape_1)
            {
                tensor.shape.rangeParallel(tensor.shape.dim(), [&tensor, &result, &reduced_dims, keepdims](const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim() ; ++i) {
                                if (reduced_dims[i]==0) {
                                        tlv.get(0)[j++]=indices[i];
                                    }else if (keepdims && (reduced_dims[i] == 1)) {
                                        tlv.get(0)[j++]=0;
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(tlv.get(0));
#pragma omp atomic
                            result.data[outputIdx]*=tensor.data[idx_linear]; 
                            }, {result.shape.dim()});
            }
            else
            {
                // 如果数据连续（对齐），则可以simd
                tensor.shape.rangeParallel(tensor.shape.dim() - 1, [&tensor, &result, &reduced_dims, keepdims](const int i, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                                               // 计算输出索引

                                               for (size_t i = 0, j = 0; i < tensor.shape.dim(); ++i)
                                               {
                                                   if (reduced_dims[i] == 0)
                                                   {
                                                       tlv.get(0)[j++] = indices[i];
                                                   }else if (keepdims && (reduced_dims[i] == 1)) {
                                                       tlv.get(0)[j++] = 0;
                                                   }
                                               }
                                               // 累加求和
                                               int outputIdx = result.shape.linearat(tlv.get(0));

                                               int shape_last = tensor.shape[-1];
                                               const ScalableTag<T> tag;
                                               const size_t lanes = Lanes(tag);
                                               size_t j = 0;
                                               T product = 1;
                                               // 前部分：处理到对齐
                                               while (j < shape_last && !IsAligned(tag, tensor.data + i + j))
                                               {
                                                   product *= tensor.data[i + j];
                                                   ++j;
                                               }

                                               // 中间部分：SIMD
                                               size_t aligned_end = shape_last - (shape_last % lanes);
                                               auto product_vec = Load(tag, tensor.data + i + j); // 初始化累乘向量
                                               j+=lanes;
                                               for (; j + lanes <= aligned_end; j += lanes)
                                               {
                                                   auto vec = Load(tag, tensor.data + i + j);
                                                   product_vec = Mul(product_vec, vec); // 向量累乘
                                               }

                                               // 将向量累乘结果写回
                                               product *= ReduceMul<T>(tag,product_vec);

                                               // 尾部分：处理剩余
                                               for (; j < shape_last; ++j)
                                               {
                                                   product *= tensor.data[i + j];
                                               }
#pragma omp atomic
                                               result.data[outputIdx] *= product; 
                                               }, {result.shape.dim()});
            }
        }
    };

    template <typename T>
    struct reducemaxDispatcher<miaobyte, T>
    {
        static void reducemax(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            const int minshape_1 = Lanes(ScalableTag<T>());
            // 如果dims的最后一个元素是tensor.shape.dim-1，则说明reducemax的数据不连续（不对齐），无法simd（需要不停跳跃）
            constant<miaobyte, T>(result, std::numeric_limits<T>::lowest());
            if (reduced_dims.rbegin()[0] == tensor.shape.dim() - 1 || tensor.shape.dim() > reduced_dims.size() || tensor.shape[-1] >= minshape_1)
            {
                tensor.shape.rangeParallel(tensor.shape.dim(), [&tensor, &result, &reduced_dims, keepdims](const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim() ; ++i) {
                                if (reduced_dims[i]==0) {
                                        tlv.get(0)[j++]=indices[i];
                                    }else if (keepdims && (reduced_dims[i] == 1)) {
                                        tlv.get(0)[j++]=0;
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(tlv.get(0));
                            result.data[outputIdx]=std::max(result.data[outputIdx],tensor.data[idx_linear]); 
                            }, {result.shape.dim()});
            }
            else
            {
                // 如果数据连续（对齐），则可以simd
                tensor.shape.rangeParallel(tensor.shape.dim() - 1, [&tensor, &result, &reduced_dims, keepdims](const int i, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                                               // 计算输出索引

                                               for (size_t i = 0, j = 0; i < tensor.shape.dim(); ++i)
                                               {
                                                   if (reduced_dims[i] == 0)
                                                   {
                                                       tlv.get(0)[j++] = indices[i];
                                                   }else if (keepdims && (reduced_dims[i] == 1)) {
                                                       tlv.get(0)[j++] =0;
                                                   }
                                               }
                                               
                                               int outputIdx = result.shape.linearat(tlv.get(0));

                                               int shape_last = tensor.shape[-1];
                                               const ScalableTag<T> tag;
                                               const size_t lanes = Lanes(tag);
                                               size_t j = 0;
                                               T maxt = tensor.data[i];
                                               // 前部分：处理到对齐
                                               while (j < shape_last && !IsAligned(tag, tensor.data + i + j))
                                               {
                                                   maxt = std::max(maxt,tensor.data[i + j]);
                                                   ++j;
                                               }

                                               // 中间部分：SIMD
                                               size_t aligned_end = shape_last - (shape_last % lanes);
                                               auto max_vec = Load(tag, tensor.data + i + j); // 初始化累乘向量为1
                                               for (; j + lanes <= aligned_end; j += lanes)
                                               {
                                                   auto vec = Load(tag, tensor.data + i + j);
                                                   max_vec = Max(max_vec, vec);  
                                               }

                                               // 将向量累乘结果写回
                                               maxt = ReduceMax(tag, max_vec);

                                               // 尾部分：处理剩余
                                               for (; j < shape_last; ++j)
                                               {
                                                   maxt = std::max(maxt,tensor.data[i + j]);
                                               }
 
                                               result.data[outputIdx] = std::max(result.data[outputIdx],maxt); 
                                               }, {result.shape.dim()});
            }
        }
    };

    template <typename T>
    struct reduceminDispatcher<miaobyte, T>
    {
        static void reducemin(const Tensor<T> &tensor, const std::vector<int> &dims, const bool keepdims, Tensor<T> &result)
        {
            std::vector<int> checkeddims = checkedDims(tensor.shape.shape, dims);
            std::vector<int> reduced_dims = reducedDim(tensor.shape.shape, checkeddims);
            const int minshape_1 = Lanes(ScalableTag<T>());
            // 如果dims的最后一个元素是tensor.shape.dim-1，则说明reducemin的数据不连续（不对齐），无法simd（需要不停跳跃）
            constant<miaobyte, T>(result, std::numeric_limits<T>::max());
            if (reduced_dims.rbegin()[0] == tensor.shape.dim() - 1 || tensor.shape.dim() > reduced_dims.size() || tensor.shape[-1] >= minshape_1)
            {
                tensor.shape.rangeParallel(tensor.shape.dim(), [&tensor, &result, &reduced_dims, keepdims](const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim() ; ++i) {
                                if (reduced_dims[i]==0) {
                                        tlv.get(0)[j++]=indices[i];
                                    }else if (keepdims && (reduced_dims[i] == 1)) {
                                        tlv.get(0)[j++]=0;
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(tlv.get(0));
 
                            result.data[outputIdx]=std::min(result.data[outputIdx],tensor.data[idx_linear]); 
                            }, {result.shape.dim()});
            }
            else
            {
                // 如果数据连续（对齐），则可以simd
                tensor.shape.rangeParallel(tensor.shape.dim() - 1, [&tensor, &result, &reduced_dims, keepdims](const int i, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                           {
                                               // 计算输出索引

                                               for (size_t i = 0, j = 0; i < tensor.shape.dim(); ++i)
                                               {
                                                   if (reduced_dims[i] == 0)
                                                   {
                                                       tlv.get(0)[j++] = indices[i];
                                                   }else if (keepdims && (reduced_dims[i] == 1)) {
                                                       tlv.get(0)[j++] = 0;
                                                   }
                                               }
                                               
                                               int outputIdx = result.shape.linearat(tlv.get(0));

                                               int shape_last = tensor.shape[-1];
                                               const ScalableTag<T> tag;
                                               const size_t lanes = Lanes(tag);
                                               size_t j = 0;
                                               T mint = tensor.data[i];
                                               // 前部分：处理到对齐
                                               while (j < shape_last && !IsAligned(tag, tensor.data + i + j))
                                               {
                                                   mint = std::min(mint,tensor.data[i + j]);
                                                   ++j;
                                               }

                                               // 中间部分：SIMD
                                               size_t aligned_end = shape_last - (shape_last % lanes);
                                               auto mint_vec = Load(tag, tensor.data + i + j); // 初始化累乘向量为1
                                               for (; j + lanes <= aligned_end; j += lanes)
                                               {
                                                   auto vec = Load(tag, tensor.data + i + j);
                                                   mint_vec = Min(mint_vec, vec);  
                                               }

                                               // 将向量累乘结果写回
                                               mint = ReduceMin(tag, mint_vec);

                                               // 尾部分：处理剩余
                                               for (; j < shape_last; ++j)
                                               {
                                                   mint = std::min(mint,tensor.data[i + j]);
                                               }
 
                                               result.data[outputIdx] = std::min(result.data[outputIdx],mint); }, {result.shape.dim()});
            }
        }
    };

}
#endif