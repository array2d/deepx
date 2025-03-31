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
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C, &scalar_op, &simd_op](int i)
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
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &b, &C, &scalar_op, &simd_op](int i)
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

    template <typename T>
    struct muladdDispatcher<miaobyte, T>
    {
        // A*B+C=D
        static void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
        {

            if (A.shape == B.shape && A.shape == C.shape && A.shape == D.shape)
            {
                D.shape.rangeParallel(D.shape.dim - 1, [&A, &B, &C, &D](int i)
                                      {
                int shape_last=D.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    D.data[i+j] = A.data[i+j] * B.data[i+j] + C.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec3 = Load(tag, C.data + i + j);
                    auto vec_result = MulAdd(vec1, vec2, vec3);
                    Store(vec_result, tag, D.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    D.data[i+j] = A.data[i+j] * B.data[i+j] + C.data[i+j];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct muladdscalarDispatcher<miaobyte, T>
    {
        // A*B*alpha+C*beta=D
        static void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
        {
            if (A.shape == B.shape && A.shape == C.shape && A.shape == D.shape)
            {
                D.shape.rangeParallel(D.shape.dim - 1, [&A, &B, &alpha, &C, &beta, &D](int i)
                                      {
                int shape_last=D.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    D.data[i+j] = alpha * A.data[i+j] * B.data[i+j] + beta * C.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto alpha_vec = Set(tag, alpha);
                    auto beta_vec = Set(tag, beta);
                    if (alpha != 1.0)
                    {
                        vec1 = Mul(vec1, alpha_vec);
                    }
                    if (beta != 0.0)
                    {
                        auto vec3 = Load(tag, C.data + i + j);
                        vec3 = Mul(vec3, beta_vec);
                        auto vec_result = MulAdd(vec1, vec2, vec3);
                        Store(vec_result, tag, D.data + i + j);
                    }else{
                        auto vec_result = Mul(vec1, vec2);
                        Store(vec_result, tag, D.data + i + j);
                    }
                   
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    D.data[i+j] = alpha * A.data[i+j] * B.data[i+j] + beta * C.data[i+j];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct mulscalaraddDispatcher<miaobyte, T>
    {
        // A*alpha+B*beta=C
        static void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim - 1, [&A, &alpha, &B, &beta, &C](int i)
                                      {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j] = alpha * A.data[i+j]   + beta * B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec_a = Load(tag, A.data + i + j);
                    auto alpha_vec = Set(tag, alpha);
                    vec_a=Mul(vec_a,alpha_vec);
                    auto vec_b = Load(tag, B.data + i + j);
                    auto beta_vec = Set(tag, beta);
                    vec_b=Mul(vec_b,beta_vec);
                    auto vec_c = Load(tag, C.data + i + j);
                    auto vec_result = Add(vec_a, vec_b);
                    Store(vec_result, tag, C.data + i + j); 
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[i+j] = alpha * A.data[i+j] + beta * B.data[i+j];
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
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

    template <typename T>
    struct divaddDispatcher<miaobyte, T>
    {
        // D= A/B+ C
        static void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
        {
            if (A.shape == B.shape && A.shape == C.shape && A.shape == D.shape)
            {
                D.shape.rangeParallel(D.shape.dim - 1, [&A, &B, &C, &D](int i)
                                      {
                    int shape_last=D.shape[-1];
                    const ScalableTag<T> tag;
                    const size_t lanes = Lanes(tag);
                    size_t j=0;

                    // 1. 处理前置未对齐部分
                    while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                        D.data[i+j] = A.data[i+j] / B.data[i+j] + C.data[i+j];
                        ++j;
                    }

                    // 2. 处理中间对齐部分
                    size_t aligned_end=shape_last-(shape_last%lanes);
                    for (; j+lanes<=aligned_end; j +=  lanes  )
                    {
                        auto vec1 = Load(tag, A.data + i + j);
                        auto vec2 = Load(tag, B.data + i + j);
                        auto vec3 = Load(tag, C.data + i + j);
                        auto vec_result = Add(Div(vec1, vec2), vec3);
                        Store(vec_result, tag, D.data + i + j);
                    }

                    // 3. 处理尾部剩余元素
                    for (;j<shape_last;j++)
                    {
                        D.data[i+j] = A.data[i+j] / B.data[i+j] + C.data[i+j];
                    } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct divscalaraddDispatcher<miaobyte, T>
    {
        //  C= A/alpha+ B/beta
        static void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim - 1, [&A, &alpha, &B, &beta, &C](int i)
                                      {
                    int shape_last=C.shape[-1];
                    const ScalableTag<T> tag;
                    const size_t lanes = Lanes(tag);
                    size_t j=0;

                    // 1. 处理前置未对齐部分
                    while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                        C.data[i+j] = A.data[i+j] / alpha + B.data[i+j] / beta;
                        ++j;
                    }

                    // 2. 处理中间对齐部分
                    size_t aligned_end=shape_last-(shape_last%lanes);
                    for (; j+lanes<=aligned_end; j +=  lanes  )
                    {
                        auto vec_a = Load(tag, A.data + i + j);
                        auto alpha_vec = Set(tag, alpha);
                        vec_a=Div(vec_a,alpha_vec);
                        auto vec_b = Load(tag, B.data + i + j);
                        auto beta_vec = Set(tag, beta);
                        vec_b=Div(vec_b,beta_vec);
                        auto vec_c = Load(tag, C.data + i + j);
                        auto vec_result = Add(vec_a, vec_b);
                        Store(vec_result, tag, C.data + i + j); 
                    }

                    // 3. 处理尾部剩余元素
                    for (;j<shape_last;j++)
                    {
                        C.data[i+j] = A.data[i+j] / alpha + B.data[i+j] / beta;
                    } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    template <typename T>
    struct divaddbetaDispatcher<miaobyte, T>
    {
        // D= A/B*alpha+ C*beta
        static void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
        {
            if (A.shape == B.shape && A.shape == C.shape && A.shape == D.shape)
            {
                D.shape.rangeParallel(D.shape.dim - 1, [&A, &alpha, &B, &beta, &C, &D](int i)
                                      {
                    int shape_last=D.shape[-1];
                    const ScalableTag<T> tag;
                    const size_t lanes = Lanes(tag);
                    size_t j=0;

                    // 1. 处理前置未对齐部分
                    while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                        D.data[i+j] = A.data[i+j] / B.data[i+j] * alpha + C.data[i+j] * beta;
                        ++j;
                    }

                    // 2. 处理中间对齐部分
                    size_t aligned_end=shape_last-(shape_last%lanes);
                    for (; j+lanes<=aligned_end; j +=  lanes  )
                    {
                        auto vec_a = Load(tag, A.data + i + j);
                        auto vec_b = Load(tag, B.data + i + j);
                        auto vec_c = Load(tag, C.data + i + j);
                        auto vec_d = Load(tag, D.data + i + j);
                        auto alpha_vec = Set(tag, alpha);
                        vec_a=Div(vec_a,vec_b);
                        vec_a=Mul(vec_a,alpha_vec);
                        auto beta_vec = Set(tag, beta);
                        vec_c=Mul(vec_c,beta_vec);
                        auto vec_result = Add(vec_a, vec_c);
                        Store(vec_result, tag, D.data + i + j); 
                    }

                    // 3. 处理尾部剩余元素
                    for (;j<shape_last;j++)
                    {
                        D.data[i+j] = A.data[i+j] / B.data[i+j] * alpha + C.data[i+j] * beta;
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
                                           {
                                               int shape_last = output.shape[-1];

                                               size_t j = 0;

                                               while (j < shape_last)
                                               {
                                                   output.data[i + j] = std::sqrt(input.data[i + j]);
                                                   ++j;
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
    struct powDispatcher<miaobyte, T>
    {
        // C=A^B
        static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
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

    template <typename T>
    struct logDispatcher<miaobyte, T>
    {
        // hwy库没有log函数，所以只能用std::log
        static void log(const Tensor<T> &input, Tensor<T> &output)
        {
            if (input.shape == output.shape)
            {
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                output.shape.rangeParallel(output.shape.dim - 1, [&input, &output](int i)
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
                C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int idx)
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
    struct maxgradDispatcher<miaobyte, T>
    {
        static void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
        {
            if (A.shape == B.shape && A.shape == output_grad.shape && A.shape == A_grad.shape && A.shape == B_grad.shape)
            {
                A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &B, &output_grad, &A_grad, &B_grad](int idx)
                                           {
                if (A.data[idx]>B.data[idx]){
                    A_grad.data[idx]=output_grad.data[idx];
                    B_grad.data[idx]=0;
                }else if (A.data[idx]<B.data[idx]){
                    A_grad.data[idx]=0;
                    B_grad.data[idx]=output_grad.data[idx];
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                    B_grad.data[idx]=output_grad.data[idx]/2;
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
                C.shape.rangeParallel(C.shape.dim - 1, [&A, b, &C](int idx)
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
    struct maxscalargradDispatcher<miaobyte, T>
    {
        static void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
        {
            if (A.shape == A_grad.shape && A.shape == output_grad.shape)
            {
                A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &b, &A_grad, &output_grad](int idx)
                                           {
                if (A.data[idx]>b){
                    A_grad.data[idx]=output_grad.data[idx];
                }else if (A.data[idx]<b){
                    A_grad.data[idx]=0;
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
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
                C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int idx)
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
    struct mingradDispatcher<miaobyte, T>
    {
        static void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
        {
            if (A.shape == B.shape && A.shape == output_grad.shape && A.shape == A_grad.shape && A.shape == B_grad.shape)
            {
                A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &B, &output_grad, &A_grad, &B_grad](int idx)
                                           {
                if (A.data[idx]<B.data[idx]){
                    A_grad.data[idx]=output_grad.data[idx];
                    B_grad.data[idx]=0;
                }else if (A.data[idx]>B.data[idx]){
                    A_grad.data[idx]=0;
                    B_grad.data[idx]=output_grad.data[idx];
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                    B_grad.data[idx]=output_grad.data[idx]/2;
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
                C.shape.rangeParallel(C.shape.dim - 1, [&A, b, &C](int idx)
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

    template <typename T>
    struct minscalargradDispatcher<miaobyte, T>
    {
        static void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
        {
            if (A.shape == A_grad.shape && A.shape == output_grad.shape)
            {
                A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &b, &A_grad, &output_grad](int idx)
                                           {
                if (A.data[idx]<b){
                    A_grad.data[idx]=output_grad.data[idx];
                }else if (A.data[idx]>b){
                    A_grad.data[idx]=0;
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                } });
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
}
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP