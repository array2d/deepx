#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "tensor.hpp"
#include "tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_common.hpp"
#include "tensorfunc/elementwise.hpp"

// ═══════════════════════════════════════════════════════════
// Helper: Metal-first, CPU-fallback pattern for binary ops
// ═══════════════════════════════════════════════════════════
#define DEEPX_METAL_DISPATCH(T, ok, kernelFn, cpuFn, A, B, C) \
    if constexpr (std::is_same_v<T, float>) ok = deepx::metal::kernels::kernelFn##_f32(A.data, B.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int8_t>) ok = deepx::metal::kernels::kernelFn##_i8(A.data, B.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int16_t>) ok = deepx::metal::kernels::kernelFn##_i16(A.data, B.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int32_t>) ok = deepx::metal::kernels::kernelFn##_i32(A.data, B.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int64_t>) ok = deepx::metal::kernels::kernelFn##_i64(A.data, B.data, C.data, A.shape.size);

#define DEEPX_METAL_UNARY_DISPATCH(T, ok, kernelFn, cpuFn, A, C) \
    if constexpr (std::is_same_v<T, float>) ok = deepx::metal::kernels::kernelFn##_f32(A.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int8_t>) ok = deepx::metal::kernels::kernelFn##_i8(A.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int16_t>) ok = deepx::metal::kernels::kernelFn##_i16(A.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int32_t>) ok = deepx::metal::kernels::kernelFn##_i32(A.data, C.data, A.shape.size); \
    else if constexpr (std::is_same_v<T, int64_t>) ok = deepx::metal::kernels::kernelFn##_i64(A.data, C.data, A.shape.size);

namespace deepx::tensorfunc
{

// ═══════════════════════════════════════════════════════════
// Binary elementwise ops: add, sub, mul, div, max, min, pow
// ═══════════════════════════════════════════════════════════

// ── add ──
template <typename T>
struct addDispatcher<miaobyte, T>
{
    static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        DEEPX_METAL_DISPATCH(T, ok, add, add_cpu, A, B, C)
        if (!ok) detail::add_cpu(A, B, C);
    }
};

// ── sub ──
template <typename T>
struct subDispatcher<miaobyte, T>
{
    static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        DEEPX_METAL_DISPATCH(T, ok, sub, sub_cpu, A, B, C)
        if (!ok) detail::sub_cpu(A, B, C);
    }
};

// ── mul ──
template <typename T>
struct mulDispatcher<miaobyte, T>
{
    static void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        DEEPX_METAL_DISPATCH(T, ok, mul, mul_cpu, A, B, C)
        if (!ok) detail::mul_cpu(A, B, C);
    }
};

// ── div ──
template <typename T>
struct divDispatcher<miaobyte, T>
{
    static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::div_f32(A.data, B.data, C.data, A.shape.size);
        if (!ok) detail::div_cpu(A, B, C);
    }
};

// ── max ──
template <typename T>
struct maxDispatcher<miaobyte, T>
{
    static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        DEEPX_METAL_DISPATCH(T, ok, max, max_cpu, A, B, C)
        if (!ok) detail::max_cpu(A, B, C);
    }
};

// ── min ──
template <typename T>
struct minDispatcher<miaobyte, T>
{
    static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        bool ok = false;
        DEEPX_METAL_DISPATCH(T, ok, min, min_cpu, A, B, C)
        if (!ok) detail::min_cpu(A, B, C);
    }
};

// ── pow (CPU-only) ──
template <typename T>
struct powDispatcher<miaobyte, T>
{
    static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        detail::assert_same_shape(A, B, C);
        detail::pow_cpu(A, B, C);
    }
};

// ═══════════════════════════════════════════════════════════
// Scalar elementwise ops
// ═══════════════════════════════════════════════════════════

// ── addscalar ──
template <typename T>
struct addscalarDispatcher<miaobyte, T>
{
    static void addscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::addscalar_cpu(A, value, C);
    }
};

// ── subscalar ──
template <typename T>
struct subscalarDispatcher<miaobyte, T>
{
    static void subscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::subscalar_cpu(A, value, C);
    }
};

// ── rsubscalar ──
template <typename T>
struct rsubscalarDispatcher<miaobyte, T>
{
    static void rsubscalar(const T value, const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::rsubscalar_cpu(value, A, C);
    }
};

// ── mulscalar ──
template <typename T>
struct mulscalarDispatcher<miaobyte, T>
{
    static void mulscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::mulscalar_cpu(A, value, C);
    }
};

// ── divscalar ──
template <typename T>
struct divscalarDispatcher<miaobyte, T>
{
    static void divscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::divscalar_cpu(A, value, C);
    }
};

// ── rdivscalar ──
template <typename T>
struct rdivscalarDispatcher<miaobyte, T>
{
    static void rdivscalar(const T value, const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::rdivscalar_cpu(value, A, C);
    }
};

// ── maxscalar ──
template <typename T>
struct maxscalarDispatcher<miaobyte, T>
{
    static void maxscalar(const Tensor<T> &A, const T b, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::maxscalar_cpu(A, b, C);
    }
};

// ── minscalar ──
template <typename T>
struct minscalarDispatcher<miaobyte, T>
{
    static void minscalar(const Tensor<T> &A, const T b, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::minscalar_cpu(A, b, C);
    }
};

// ── powscalar ──
template <typename T>
struct powscalarDispatcher<miaobyte, T>
{
    static void powscalar(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::powscalar_cpu(A, value, C);
    }
};

// ── rpowscalar ──
template <typename T>
struct rpowscalarDispatcher<miaobyte, T>
{
    static void rpowscalar(const T value, const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::rpowscalar_cpu(value, A, C);
    }
};

// ═══════════════════════════════════════════════════════════
// Unary elementwise ops (Metal + CPU fallback)
// ═══════════════════════════════════════════════════════════

// ── sqrt ──
template <typename T>
struct sqrtDispatcher<miaobyte, T>
{
    static void sqrt(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::sqrt_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::sqrt_cpu(A, C);
    }
};

// ── exp ──
template <typename T>
struct expDispatcher<miaobyte, T>
{
    static void exp(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::exp_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::exp_cpu(A, C);
    }
};

// ── log ──
template <typename T>
struct logDispatcher<miaobyte, T>
{
    static void log(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::log_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::log_cpu(A, C);
    }
};

// ── sin ──
template <typename T>
struct sinDispatcher<miaobyte, T>
{
    static void sin(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::sin_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::sin_cpu(A, C);
    }
};

// ── cos ──
template <typename T>
struct cosDispatcher<miaobyte, T>
{
    static void cos(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::cos_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::cos_cpu(A, C);
    }
};

// ── tan ──
template <typename T>
struct tanDispatcher<miaobyte, T>
{
    static void tan(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        bool ok = false;
        if constexpr (std::is_same_v<T, float>)
            ok = deepx::metal::kernels::tan_f32(A.data, C.data, A.shape.size);
        if (!ok) detail::tan_cpu(A, C);
    }
};

// ── invert ──
template <typename T>
struct invertDispatcher<miaobyte, T>
{
    static void invert(const Tensor<T> &A, Tensor<T> &C)
    {
        detail::assert_same_shape(A, C);
        detail::invert_cpu(A, C);
    }
};

// Specialization for bool
template <>
struct invertDispatcher<miaobyte, bool>
{
    static void invert(const Tensor<bool> &A, Tensor<bool> &C)
    {
        detail::assert_same_shape(A, C);
        detail::invert_cpu(A, C);
    }
};

// ═══════════════════════════════════════════════════════════
// Comparison ops (CPU-only)
// ═══════════════════════════════════════════════════════════

// ── equal ──
template <typename T, typename MaskT>
struct equalDispatcher<miaobyte, T, MaskT>
{
    static void equal(const Tensor<T> &A, const Tensor<T> &B, float epsilon, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, B, mask);
        if (epsilon == 0) {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = A.data[i] == B.data[i];
        } else {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = std::abs(static_cast<double>(A.data[i]) - static_cast<double>(B.data[i])) <= static_cast<double>(epsilon);
        }
    }
};

// ── equalscalar ──
template <typename T, typename MaskT>
struct equalscalarDispatcher<miaobyte, T, MaskT>
{
    static void equalscalar(const Tensor<T> &A, const T scalar, float epsilon, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, mask);
        if (epsilon == 0) {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = A.data[i] == scalar;
        } else {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = std::abs(static_cast<double>(A.data[i]) - static_cast<double>(scalar)) <= static_cast<double>(epsilon);
        }
    }
};

// ── notequal ──
template <typename T, typename MaskT>
struct notequalDispatcher<miaobyte, T, MaskT>
{
    static void notequal(const Tensor<T> &A, const Tensor<T> &B, float epsilon, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, B, mask);
        if (epsilon == 0) {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = A.data[i] != B.data[i];
        } else {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = std::abs(static_cast<double>(A.data[i]) - static_cast<double>(B.data[i])) > static_cast<double>(epsilon);
        }
    }
};

// ── notequalscalar ──
template <typename T, typename MaskT>
struct notequalscalarDispatcher<miaobyte, T, MaskT>
{
    static void notequalscalar(const Tensor<T> &A, const T scalar, float epsilon, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, mask);
        if (epsilon == 0) {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = A.data[i] != scalar;
        } else {
            for (int64_t i = 0; i < A.shape.size; ++i)
                mask.data[i] = std::abs(static_cast<double>(A.data[i]) - static_cast<double>(scalar)) > static_cast<double>(epsilon);
        }
    }
};

// ── less ──
template <typename T, typename MaskT>
struct lessDispatcher<miaobyte, T, MaskT>
{
    static void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, B, mask);
        for (int64_t i = 0; i < A.shape.size; ++i)
            mask.data[i] = A.data[i] < B.data[i];
    }
};

// ── lessscalar ──
template <typename T, typename MaskT>
struct lessscalarDispatcher<miaobyte, T, MaskT>
{
    static void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, mask);
        for (int64_t i = 0; i < A.shape.size; ++i)
            mask.data[i] = A.data[i] < scalar;
    }
};

// ── greater ──
template <typename T, typename MaskT>
struct greaterDispatcher<miaobyte, T, MaskT>
{
    static void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, B, mask);
        for (int64_t i = 0; i < A.shape.size; ++i)
            mask.data[i] = A.data[i] > B.data[i];
    }
};

// ── greaterscalar ──
template <typename T, typename MaskT>
struct greaterscalarDispatcher<miaobyte, T, MaskT>
{
    static void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
    {
        detail::assert_same_shape(A, mask);
        for (int64_t i = 0; i < A.shape.size; ++i)
            mask.data[i] = A.data[i] > scalar;
    }
};

} // namespace deepx::tensorfunc

#undef DEEPX_METAL_DISPATCH
#undef DEEPX_METAL_UNARY_DISPATCH

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
