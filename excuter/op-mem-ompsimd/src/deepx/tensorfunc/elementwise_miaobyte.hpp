#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP

#include <random>
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
            C.shape.rangeElementwiseParallel([&A, &B, &C, &scalar_op, &simd_op](int i, int i_end)
                                             {
                                      
                                      const ScalableTag<T> tag;
                                      const size_t lanes = Lanes(tag);
                                      size_t j = 0;

                                      // 1. 处理前置未对齐部分
                                      while (j < i_end && !IsAligned(tag, A.data + i + j))
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], B.data[i + j], c);
                                          C.data[i + j] = c;
                                          ++j;
                                      }

                                      // 2. 处理中间对齐部分
                                      size_t aligned_end = i_end - (i_end % lanes);
                                      for (; j + lanes <= aligned_end; j += lanes)
                                      {
                                          simd_op(A.data + i + j, B.data + i + j, C.data + i + j, lanes);
                                      }

                                      // 3. 处理尾部剩余元素
                                      for (; j < i_end; j++)
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
            C.shape.rangeElementwiseParallel([&A, &b, &C, &scalar_op, &simd_op](int i, int i_end)
                                             {
                                      const ScalableTag<T> tag;
                                      const size_t lanes = Lanes(tag);
                                      size_t j = 0;

                                      // 1. 处理前置未对齐部分
                                      while (j < i_end && !IsAligned(tag, A.data + i + j))
                                      {
                                          T c;
                                          scalar_op(A.data[i + j], b, c);
                                          C.data[i + j] = c;
                                          ++j;
                                      }

                                      // 2. 处理中间对齐部分
                                      size_t aligned_end = i_end - (i_end % lanes);
                                      for (; j + lanes <= aligned_end; j += lanes)
                                      {
                                          simd_op(A.data + i + j, b, C.data + i + j, lanes);
                                      }

                                      // 3. 处理尾部剩余元素
                                      for (; j < i_end; j++)
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

    // todtype
    template <typename T, typename Dtype>
    static void todtype(const Tensor<T> &A, Tensor<Dtype> &C)
    {
        C.shape.rangeElementwiseParallel([&A, &C](int i, int i_end)
                                         {
            for (int j = 0; j < i_end; j++)
            {
                C.data[i + j] = static_cast<Dtype>(A.data[i + j]);
            } });
    }

    // add
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
                A.shape.rangeElementwiseParallel([&A, &C](int idx, int idx_end)
                                                 {
                                           for (int j=0;j<idx_end;j++)
                                           {
                                                C.data[idx+j]=~A.data[idx+j];
                                           } });
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::sqrt(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Sqrt(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      {
                                               size_t j = 0;

                                               while (j < i_end)
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
                C.shape.rangeElementwiseParallel([&A, &B, &C](int i, int i_end)
                                                 {
                                         for (int j = 0; j < i_end; j++)
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
                output.shape.rangeElementwiseParallel([&input, &output, &value](int i, int i_end)
                                                      {
                                             for (int j = 0; j < i_end; j++)
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
                output.shape.rangeElementwiseParallel([&input, &output, &value](int i, int i_end)
                                                      {
                                                for (int j = 0; j < i_end; j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      { for (int j = 0; j < i_end; j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      { for (int j = 0; j < i_end; j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::sin(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Sin(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::cos(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Cos(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
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
                output.shape.rangeElementwiseParallel([&input, &output](int i, int i_end)
                                                      {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::tan(input.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto vec_result = Tan(vec);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
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
                C.shape.rangeElementwiseParallel([&A, &B, &C](int i, int i_end)
                                                 {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;
 
                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j]=std::max(A.data[i+j],B.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + i + j);  // 加载数组2的向量
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + i + j); // 存储结果向量
                }  

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
                {
                    C.data[i+j]=std::max(A.data[i+j],B.data[i+j]);
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
                C.shape.rangeElementwiseParallel([&A, b, &C](int i, int i_end)
                                                 {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < i_end && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j]=std::max(A.data[i+j],b);
                    ++j;
                }   

                // 2. 处理中间对齐部分
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);   
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + i + j); // 存储结果向量
                }   

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
                {
                    C.data[i+j]=std::max(A.data[i+j],b);
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
                C.shape.rangeElementwiseParallel([&A, &B, &C](int i, int i_end)
                                                 {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分    
                while (j < i_end && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j]=std::min(A.data[i+j],B.data[i+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + i + j);  // 加载数组2的向量
                    auto vec_result = Min(vec1, vec2);  // 向量比较 
                    Store(vec_result, tag, C.data + i + j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++)
                {
                    C.data[i+j]=std::min(A.data[i+j],B.data[i+j]);
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
                C.shape.rangeElementwiseParallel([&A, b, &C](int i, int i_end)
                                                 {   
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分        
                while (j < i_end && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j]=std::min(A.data[i+j],b);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=i_end-(i_end%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);       
                    auto vec_result = Min(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + i + j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<i_end;j++){
                    C.data[i+j]=std::min(A.data[i+j],b);
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // equal
    template <typename T, typename MaskT>
    struct equalDispatcher<miaobyte, T, MaskT>
    {
        static void equal(const Tensor<T> &A, const Tensor<T> &B, const float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &B, &mask, epsilon](int i, int i_end)
                                                 {
                                            for (int j = 0; j < i_end; j++)
                                            {
                                                if (epsilon == 0)
                                                {
                                                    mask.data[i+j]=A.data[i+j]==B.data[i+j];
                                                }
                                                else{
                                                    mask.data[i+j]=std::abs(A.data[i+j]-B.data[i+j])<=epsilon;
                                                }
                                            } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // equalscalar
    template <typename T, typename MaskT>
    struct equalscalarDispatcher<miaobyte, T, MaskT>
    {
        static void equalscalar(const Tensor<T> &A, const T scalar, const float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &mask, &scalar, epsilon](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {
                    if (epsilon == 0)
                    {
                        mask.data[i+j]=A.data[i+j]==scalar;
                    }
                    else{
                        mask.data[i+j]=std::abs(A.data[i+j]-scalar)<=epsilon;
                    }
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        };
    };

    // less
    template <typename T, typename MaskT>
    struct lessDispatcher<miaobyte, T, MaskT>
    {
        static void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &B, &mask](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {
                    mask.data[i+j]=A.data[i+j]<B.data[i+j];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // lessscalar
    template <typename T, typename MaskT>
    struct lessscalarDispatcher<miaobyte, T, MaskT>
    {
        static void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &mask, &scalar](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {
                    mask.data[i+j]=A.data[i+j]<scalar;
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // greater
    template <typename T, typename MaskT>
    struct greaterDispatcher<miaobyte, T, MaskT>
    {
        static void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape == B.shape && mask.shape == A.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &B, &mask](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {
                    mask.data[i+j]=A.data[i+j]>B.data[i+j];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // greaterscalar
    template <typename T, typename MaskT>
    struct greaterscalarDispatcher<miaobyte, T, MaskT>
    {
        static void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape == mask.shape)
            {
                A.shape.rangeElementwiseParallel([&A, &mask, &scalar](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {
                    mask.data[i+j]=A.data[i+j]>scalar;
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // switch
    template <typename T, typename casesT>
    struct switchDispatcher<miaobyte, T, casesT>
    {
        static void Switch(const vector<Tensor<T> *> tensors, const Tensor<casesT> &cases, Tensor<T> &C)
        {
            if (cases.shape == C.shape)
            {
                C.shape.rangeElementwiseParallel([&tensors, &cases, &C](int i, int i_end)
                                                 {
                for (int j = 0; j < i_end; j++)
                {   
                    int which_tensor=cases.data[i];
                    C.data[i+j]=tensors[which_tensor]->data[i];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // dropout
    template <typename T>
    struct dropoutDispatcher<miaobyte, T>
    {
        static void dropout(const Tensor<T> &A, const float p, const unsigned int seed, Tensor<T> &C)
        {
            if (A.shape == C.shape)
            {
                std::uniform_real_distribution<double> distribution(0, 1);
                std::default_random_engine generator;
                if (seed != 0)
                {
                    generator.seed(seed);
                }
                else
                {
                    std::random_device rd;
                    generator.seed(rd());
                }

                A.shape.rangeElementwiseParallel([&A, &C, &p, &distribution, &generator](int i, int i_end)
                                                 {
                                        for (int j = 0; j < i_end; j++)
                                        {
                                            double rand = distribution(generator);
                                            if (rand < p)
                                            {
                                                C.data[i+j]=0;
                                            }
                                            else
                                            {
                                                C.data[i+j]=A.data[i+j];
                                            }
                                        } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
};
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP