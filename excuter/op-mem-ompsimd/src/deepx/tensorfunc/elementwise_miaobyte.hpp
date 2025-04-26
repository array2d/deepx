#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP

#include <cblas.h>
#include <cmath>
#include <hwy/highway.h>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    using namespace hwy::HWY_NAMESPACE;
    // 通用元素级操作模板
    template <typename T, typename ScalarOpFunc, typename SimdOpFunc>
    void elementwise_A_B_C(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C,
                           ScalarOpFunc scalar_op, SimdOpFunc simd_op)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim() - 1, [&A, &B, &C, &scalar_op, &simd_op](int i)
                                  {
                                      int shape_last = C.shape[-1];
                                      const ScalableTag<T> tag;
                                      const size_t lanes = Lanes(tag);
                                      size_t j = 0;

                                      // 1. 处理前置未对齐部分
                                      while (j < shape_last && !IsAligned(tag, A.data + i + j))
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], B.data[i + j], c);
                                          C.data[i + j] = c;
                                          ++j;
                                      }

                                      // 2. 处理中间对齐部分
                                      size_t aligned_end = shape_last - (shape_last % lanes);
                                      for (; j + lanes <= aligned_end; j += lanes)
                                      {
                                          simd_op(A.data + i + j, B.data + i + j, C.data + i + j, lanes);
                                      }

                                      // 3. 处理尾部剩余元素
                                      for (; j < shape_last; j++)
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], B.data[i + j], c);
                                          C.data[i + j] = c;
                                      } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // 通用元素级操作模板
    template <typename T, typename ScalarOpFunc, typename SimdOpFunc>
    void elementwise_A_b_C(const Tensor<T> &A, const T b, Tensor<T> &C,
                           ScalarOpFunc scalar_op, SimdOpFunc simd_op)
    {
        if (A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim() - 1, [&A, &b, &C, &scalar_op, &simd_op](int i)
                                  {
                                      int shape_last = C.shape[-1];
                                      const ScalableTag<T> tag;
                                      const size_t lanes = Lanes(tag);
                                      size_t j = 0;

                                      // 1. 处理前置未对齐部分
                                      while (j < shape_last && !IsAligned(tag, A.data + i + j))
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], b, c);
                                          C.data[i + j] = c;
                                          ++j;
                                      }

                                      // 2. 处理中间对齐部分
                                      size_t aligned_end = shape_last - (shape_last % lanes);
                                      for (; j + lanes <= aligned_end; j += lanes)
                                      {
                                          simd_op(A.data + i + j, b, C.data + i + j, lanes);
                                      }

                                      // 3. 处理尾部剩余元素
                                      for (; j < shape_last; j++)
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], b, c);
                                          C.data[i + j] = c;
                                      } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // 通用实现
    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {

            elementwise_A_B_C<T>(A, B, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a + b; },
                                 // SIMD操作
                                 [](const T *a, const T *b, T *c, size_t size)
                                 {
                const ScalableTag<T> tag;
                auto vec1 = Load(tag, a);
                auto vec2 = Load(tag, b);
                auto vec_result = Add(vec1, vec2);
                Store(vec_result, tag, c); });
        }
    };

    template <typename T>
    struct addscalarDispatcher<miaobyte, T>
    {
        static void addscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
        {
            elementwise_A_b_C<T>(A, value, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a + b; },
                                 // SIMD操作
                                 [](const T *a, const T b, T *c, size_t size)
                                 {
                const ScalableTag<T> tag;
                auto vec1 = Load(tag, a);
                auto scalar = Set(tag, b);
                auto vec_result = Add(vec1, scalar);
                Store(vec_result, tag, c); });
        }
    };

    // 添加 sub 的模板特化实现
    template <typename T>
    struct subDispatcher<miaobyte, T>
    {
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            elementwise_A_B_C<T>(A, B, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a - b; },
                                 // SIMD操作
                                 [](const T *a, const T *b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto vec2 = Load(tag, b);
                                    auto vec_result = Sub(vec1, vec2);
                                    Store(vec_result, tag, c); });
        }
    };

    template <typename T>
    struct subscalarDispatcher<miaobyte, T>
    {
        static void subscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
        {
            elementwise_A_b_C<T>(A, value, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a - b; },
                                 // SIMD操作
                                 [](const T *a, const T b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto scalar = Set(tag, b);
                                    auto vec_result = Sub(vec1, scalar);
                                    Store(vec_result, tag, c); });
        }
    };

    // 添加 mul 的模板特化实现
    template <typename T>
    struct mulDispatcher<miaobyte, T>
    {
        static void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            elementwise_A_B_C<T>(A, B, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a * b; },
                                 // SIMD操作
                                 [](const T *a, const T *b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto vec2 = Load(tag, b);
                                    auto vec_result = Mul(vec1, vec2);
                                    Store(vec_result, tag, c); });
        }
    };

    template <typename T>
    struct mulscalarDispatcher<miaobyte, T>
    {
        static void mulscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
        {
            elementwise_A_b_C<T>(A, value, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a * b; },
                                 // SIMD操作
                                 [](const T *a, const T b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto scalar = Set(tag, b);
                                    auto vec_result = Mul(vec1, scalar);
                                    Store(vec_result, tag, c); });
        }
    };

    // 添加 div 的模板特化实现
    template <typename T>
    struct divDispatcher<miaobyte, T>
    {
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            elementwise_A_B_C<T>(A, B, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a / b; },
                                 // SIMD操作
                                 [](const T *a, const T *b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto vec2 = Load(tag, b);
                                    auto vec_result = Div(vec1, vec2);
                                    Store(vec_result, tag, c); });
        }
    };

    template <typename T>
    struct divscalarDispatcher<miaobyte, T>
    {
        static void divscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
        {
            elementwise_A_b_C<T>(A, value, C,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = a / b; },
                                 // SIMD操作
                                 [](const T *a, const T b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto scalar = Set(tag, b);
                                    auto vec_result = Div(vec1, scalar);
                                    Store(vec_result, tag, c); });
        }
    };

    template <typename T>
    struct rdivscalarDispatcher<miaobyte, T>
    {
        static void rdivscalar(const T value, const Tensor<T> &In, Tensor<T> &Out)
        {
            elementwise_A_b_C<T>(In, value, Out,
                                 // 标量操作
                                 [](const T &a, const T &b, T &c)
                                 { c = b / a; },
                                 // SIMD操作
                                 [](const T *a, const T b, T *c, size_t size)
                                 {
                                    const ScalableTag<T> tag;
                                    auto vec1 = Load(tag, a);
                                    auto scalar = Set(tag, b);
                                    auto vec_result = Div(scalar, vec1);
                                    Store(vec_result, tag, c); });
        }
    };

    // invert
    template <typename T>
    struct invertDispatcher<miaobyte, T>
    {
        static void invert(const Tensor<T> &A, Tensor<T> &C)
        {   
            if (A.shape == C.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &C](int idx)
                                      {
                                           for (int j=0;j<A.shape[-1];j++)
                                           {
                                                C.data[idx+j]=~A.data[idx+j];
                                           } 
                                      });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };  

    template <typename T>
    struct sqrtDispatcher<miaobyte, T, std::enable_if_t<std::is_floating_point_v<T>>>
    {
        static void sqrt(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::sqrt(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Sqrt(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = std::sqrt(input.data[i+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
    template <typename T>
    struct sqrtDispatcher<miaobyte, T, std::enable_if_t<std::is_integral_v<T>>>
    {
        static void sqrt(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           {
                                               int shape_last = output.shape[-1];

                                               size_t j = 0;

                                               while (j < shape_last)
                                               {
                                                   output.data[i + j] = std::sqrt(input.data[i + j]);
                                                   ++j;
                                               } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct powDispatcher<miaobyte, T>
    {
        // C=A^B
        static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim() - 1, [&A, &B, &C](int i)
                                      {
                                         for (int j = 0; j < C.shape[-1]; j++)
                                         C.data[i+j] = std::pow(A.data[i+j], B.data[i+j]); });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct powscalarDispatcher<miaobyte, T>
    {
        // C=A^value
        //  highway 不支持POW
        static void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output, &value](int i)
                                           {
                                             for (int j = 0; j < output.shape[-1]; j++)
                                                output.data[i+j] = std::pow(input.data[i+j], value); });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // rpowscalar
    template <typename T>
    struct rpowscalarDispatcher<miaobyte, T>
    {
        static void rpowscalar(const T value, const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output, &value](int i)
                                           {
                                                for (int j = 0; j < output.shape[-1]; j++)
                                                output.data[i+j] = std::pow(value, input.data[i+j]); });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };  

    template <typename T>
    struct logDispatcher<miaobyte, T>
    {
        // hwy库没有log函数，所以只能用std::log
        static void log(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           { for (int j = 0; j < output.shape[-1]; j++)
                                                output.data[i+j] = std::log(input.data[i+j]); });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct expDispatcher<miaobyte, T>
    {
        // 发现hwy库没有exp函数，所以只能用std::exp
        static void exp(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           { for (int j = 0; j < output.shape[-1]; j++)
                                                output.data[i+j] = std::exp(input.data[i+j]); });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct sinDispatcher<miaobyte, T>
    {

        static void sin(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::sin(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Sin(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = std::sin(input.data[i+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct cosDispatcher<miaobyte, T>
    {

        static void cos(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::cos(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Cos(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = std::cos(input.data[i+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct tanDispatcher<miaobyte, T>
    {

        static void tan(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim() - 1, [&input, &output](int i)
                                           {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::tan(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Tan(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = std::tan(input.data[i+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct maxDispatcher<miaobyte, T>
    {
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim() - 1, [&A, &B, &C](int idx)
                                      {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;
 
                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::max(A.data[idx+j],B.data[idx+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + idx+j);  // 加载数组2的向量
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }  

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::max(A.data[idx+j],B.data[idx+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct maxscalarDispatcher<miaobyte, T>
    {
        static void maxscalar(const Tensor<T> &A, const T b, Tensor<T> &C)
        {
            if (A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim() - 1, [&A, b, &C](int idx)
                                      {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::max(A.data[idx+j],b);
                    ++j;
                }   

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);   
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }   

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::max(A.data[idx+j],b);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct minDispatcher<miaobyte, T>
    {
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim() - 1, [&A, &B, &C](int idx)
                                      {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分    
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::min(A.data[idx+j],B.data[idx+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + idx+j);  // 加载数组2的向量
                    auto vec_result = Min(vec1, vec2);  // 向量比较 
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::min(A.data[idx+j],B.data[idx+j]);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct minscalarDispatcher<miaobyte, T>
    {
        static void minscalar(const Tensor<T> &A, const T b, Tensor<T> &C)
        {
            if (A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim() - 1, [&A, b, &C](int idx)
                                      {   
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分        
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::min(A.data[idx+j],b);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);       
                    auto vec_result = Min(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++){
                    C.data[idx+j]=std::min(A.data[idx+j],b);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    //equal
    template <typename T,typename MaskT>
    struct equalDispatcher<miaobyte, T,MaskT>
    {
        static void equal(const Tensor<T> &A, const Tensor<T> &B,const float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {   
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &B, &mask,epsilon](int idx)
                                      {
                                            for (int i = 0; i < A.shape[-1]; i++)
                                            {
                                                if (epsilon == 0)
                                                {
                                                    mask.data[idx+i]=A.data[idx+i]==B.data[idx+i];
                                                }
                                                else{
                                                    mask.data[idx+i]=std::abs(A.data[idx+i]-B.data[idx+i])<=epsilon;
                                                }
                                            }
                                            });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    //equalscalar
    template <typename T,typename MaskT>
    struct equalscalarDispatcher<miaobyte, T,MaskT>
    {
        static void equalscalar(const Tensor<T> &A, const T scalar,const float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &mask, &scalar,epsilon](int idx)
                                      {
                for (int i = 0; i < A.shape[-1]; i++)
                {
                    if (epsilon == 0)
                    {
                        mask.data[idx+i]=A.data[idx+i]==scalar;
                    }
                    else{
                        mask.data[idx+i]=std::abs(A.data[idx+i]-scalar)<=epsilon;
                    }
                }
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        };
    };

    //less
    template <typename T,typename MaskT>
    struct lessDispatcher<miaobyte, T,MaskT>
    {
        static void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &B, &mask](int idx)
                                      {
                for (int i = 0; i < A.shape[-1]; i++)
                {
                    mask.data[idx+i]=A.data[idx+i]<B.data[idx+i];
                }   
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }   
    };

    //lessscalar
    template <typename T,typename MaskT>
    struct lessscalarDispatcher<miaobyte, T,MaskT>
    {
        static void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &mask, &scalar](int idx)
                                      {
                for (int i = 0; i < A.shape[-1]; i++)
                {
                    mask.data[idx+i]=A.data[idx+i]<scalar;
                }
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }   
    };
    
    //greater
    template <typename T,typename MaskT>
    struct greaterDispatcher<miaobyte, T,MaskT>
    {
        static void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &B, &mask](int idx)
                                      {
                for (int i = 0; i < A.shape[-1]; i++)
                {
                    mask.data[idx+i]=A.data[idx+i]>B.data[idx+i];
                }
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    //greaterscalar
    template <typename T,typename MaskT>
    struct greaterscalarDispatcher<miaobyte, T,MaskT>
    {
        static void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeParallel(A.shape.dim()-1, [&A, &mask, &scalar](int idx)
                                      {
                for (int i = 0; i < A.shape[-1]; i++)
                {
                    mask.data[idx+i]=A.data[idx+i]>scalar;
                }
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }   
    };      

    //switch
    template <typename T,typename casesT>
    struct switchDispatcher<miaobyte, T,casesT>
    {
        static void Switch(const vector<Tensor<T>*> tensors,const Tensor<casesT> &cases, Tensor<T> &C)
        {
            if (cases.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim()-1, [&tensors, &cases, &C](int idx)
                                      {
                for (int i = 0; i < C.shape[-1]; i++)
                {   
                    int which_tensor=cases.data[idx];
                    C.data[idx+i]=tensors[which_tensor]->data[idx];
                }
                });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }   
        }
    };      
    
};
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP