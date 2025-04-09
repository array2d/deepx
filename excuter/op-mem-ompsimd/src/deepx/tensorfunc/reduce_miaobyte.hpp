#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <hwy/highway.h>

#include <deepx/vector_combination.hpp>
#include "deepx/tensor.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"

namespace deepx::tensorfunc
{
    using namespace hwy::HWY_NAMESPACE;

    // 通用元素级操作模板
    template <typename T, typename ScalarOpFunc, typename SimdOpFunc, typename SimdReduceOpFunc>
    void reduce_A_B_C(const Tensor<T> &input, const std::vector<int> &dims, Tensor<T> &output, const bool keepdims, ScalarOpFunc scalar_op, SimdOpFunc simd_op, SimdReduceOpFunc simd_reduce_op)
    {
        std::vector<int> checkeddims = checkedDims(input.shape.shape, dims);
        std::vector<int> reduced_dims = reducedDim(input.shape.shape, checkeddims);
        const ScalableTag<T> _tag;
        size_t minshape_1 = Lanes(_tag);
        // if (true)
        if (reduced_dims.rbegin()[0] == input.shape.dim - 1 || input.shape.dim > reduced_dims.size() || input.shape[-1] >= minshape_1)
        {
            input.shape.rangeParallel(input.shape.dim, [&input, &output, &reduced_dims, keepdims, &scalar_op](const int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                      {
                                          // 计算输出索引

                                          for (size_t i = 0, j = 0; i < input.shape.dim; ++i)
                                          {
                                              if (reduced_dims[i] == 0)
                                              {
                                                  newIndices[j++] = indices[i];
                                              }
                                              else if (keepdims && (reduced_dims[i] == 1))
                                              {
                                                  newIndices[j++] = indices[i];
                                              }
                                          }

                                          int outputIdx = output.shape.linearat(newIndices);

                                          scalar_op(input.data[idx_linear], output.data[outputIdx]); }, output.shape.dim);
        }
        else
        {
            // 这里有bug，todo
            //  如果数据连续（对齐），则可以simd
            input.shape.rangeParallel(input.shape.dim - 1, [&input, &output, &reduced_dims, keepdims, &scalar_op, &simd_op, &simd_reduce_op](const int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                      {
                                          // 计算输出索引
                                          for (size_t i = 0, j = 0; i < input.shape.dim; ++i)
                                          {
                                              if (reduced_dims[i] == 0)
                                              {
                                                  newIndices[j++] = indices[i];
                                              }
                                              else if (keepdims && (reduced_dims[i] == 1))
                                              {
                                                  newIndices[j++] = indices[i];
                                              }
                                          }
                                          int outputIdx = output.shape.linearat(newIndices);

                                          int shape_last = input.shape[-1];
                                          const ScalableTag<T> tag;
                                          const size_t lanes = Lanes(tag);
                                          size_t j = 0;

                                          // 前部分：处理到对齐
                                          while (j < shape_last && !IsAligned(tag, input.data + idx_linear + j))
                                          {
                                              scalar_op(input.data[idx_linear + j], output.data[outputIdx] );
                                              ++j;
                                          }

                                          // 中间部分：SIMD
                                          
                                          size_t aligned_end = shape_last - (shape_last % lanes);
                                          auto simd_vec = Zero(tag);
                                          for (; j + lanes <= aligned_end; j += lanes)
                                          {
                                              simd_op(input.data + idx_linear + j, simd_vec);
                                          }

                                          // 将向量累加结果写回
                                          simd_reduce_op(simd_vec, output.data[outputIdx]);

                                          // 尾部分：处理剩余
                                          for (; j < shape_last; ++j)
                                          {
                                              scalar_op(input.data[idx_linear + j], output.data[outputIdx] );
                                          } }, output.shape.dim);
        }
    }

    template <typename T>
    struct sumDispatcher<miaobyte, T>
    {
        static void sum(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result, const bool keepdims)
        {
            constant<miaobyte, T>(result, T(0));
            reduce_A_B_C<T>(tensor, dims, result, keepdims,
                            [](const T &a, T &b)
                            {
                            #pragma omp atomic
                            b = a + b; 
                            },
                             [](const T *a, auto &b_vec)
                            {
                               const ScalableTag<T> tag;
                               auto a_vec = Load(tag, a);
                               b_vec = Add(a_vec, b_vec);
                             }, 
                            [](auto b_vec, T &b)
                            {
                               const ScalableTag<T> tag;
                                #pragma omp atomic
                               b += ReduceSum(tag, b_vec); 
                               });
        };
    };
    template <typename T>
    struct prodDispatcher<miaobyte, T>
    {
        static void prod(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result, const bool keepdims)
        {

            std::vector<int> sorted_dims = dims;
            if (dims.size() == 0)
            {
                sorted_dims = arrange(tensor.shape.dim);
            }
            // 从大到小排序
            std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
            std::vector<int> sumMap = reduceDimMap(tensor.shape, sorted_dims);
            // 如果dims的最后一个元素是tensor.shape.dim-1，则说明求和的数据不连续（不对齐），无法simd（需要不停跳跃）
            constant(result, T(1));
            if (sorted_dims.at(sorted_dims.size() - 1) == tensor.shape.dim - 1 && tensor.shape.dim > sorted_dims.size())
            {
                tensor.shape.rangeParallel(tensor.shape.dim, [&tensor, &result, &sumMap](const int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                           {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim ; ++i) {
                                if (sumMap[i]==0) {
                                        newIndices[j++]=indices[i];
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(newIndices);
#pragma omp atomic
                            result.data[outputIdx]*=tensor.data[idx_linear]; }, result.shape.dim);
            }
            else
            {
                // 如果数据连续（对齐），则可以simd
                tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &result, &sumMap](const int i, const std::vector<int> &indices, std::vector<int> &newIndices)
                                           {
                                               // 计算输出索引

                                               for (size_t i = 0, j = 0; i < tensor.shape.dim; ++i)
                                               {
                                                   if (sumMap[i] == 0)
                                                   {
                                                       newIndices[j++] = indices[i];
                                                   }
                                               }
                                               // 累加求和
                                               int outputIdx = result.shape.linearat(newIndices);

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
                                               auto product_vec = One(tag); // 初始化累乘向量为1
                                               for (; j + lanes <= aligned_end; j += lanes)
                                               {
                                                   auto vec = Load(tag, tensor.data + i + j);
                                                   product_vec = Mul(product_vec, vec); // 向量累乘
                                               }

                                               // 将向量累乘结果写回
                                               product *= ReduceMul(tag, product_vec);

                                               // 尾部分：处理剩余
                                               for (; j < shape_last; ++j)
                                               {
                                                   product *= tensor.data[i + j];
                                               }
#pragma omp atomic
                                               result.data[outputIdx] *= product; }, result.shape.dim);
            }
        }
    };
}
#endif