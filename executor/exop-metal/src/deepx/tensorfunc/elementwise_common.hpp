#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP

#if defined(__APPLE__)
  #include <TargetConditionals.h>
#endif

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/metal_common.hpp"

namespace deepx::tensorfunc::detail
{
    template <typename T>
    inline void assert_same_shape(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C)
    {
        if (A.shape.size != B.shape.size || A.shape.size != C.shape.size ||
            A.shape.shape != B.shape.shape || A.shape.shape != C.shape.shape)
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    inline void assert_same_shape(const Tensor<T> &A, const Tensor<T> &C)
    {
        if (A.shape.size != C.shape.size ||
            A.shape.shape != C.shape.shape)
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // ── CPU fallback implementations ──

    template <typename T>
    inline void add_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] + B.data[i];
    }

    template <typename T>
    inline void sub_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] - B.data[i];
    }

    template <typename T>
    inline void mul_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] * B.data[i];
    }

    template <typename T>
    inline void div_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] / B.data[i];
    }

    template <typename T>
    inline void max_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] > B.data[i] ? A.data[i] : B.data[i];
    }

    template <typename T>
    inline void min_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] < B.data[i] ? A.data[i] : B.data[i];
    }

    template <typename T>
    inline void relu_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        T zero = T(0);
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] > zero ? A.data[i] : zero;
    }

    template <typename T>
    inline void neg_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = -A.data[i];
    }

    template <typename T>
    inline void abs_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] < T(0) ? -A.data[i] : A.data[i];
    }

    template <typename T>
    inline void addscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] + v;
    }

    template <typename T>
    inline void subscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] - v;
    }

    template <typename T>
    inline void rsubscalar_cpu(const T v, const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = v - A.data[i];
    }

    template <typename T>
    inline void mulscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] * v;
    }

    template <typename T>
    inline void divscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] / v;
    }

    template <typename T>
    inline void rdivscalar_cpu(const T v, const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = v / A.data[i];
    }

    template <typename T>
    inline void maxscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] > v ? A.data[i] : v;
    }

    template <typename T>
    inline void minscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = A.data[i] < v ? A.data[i] : v;
    }

    template <typename T>
    inline void pow_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::pow(A.data[i], B.data[i]);
    }

    template <typename T>
    inline void powscalar_cpu(const Tensor<T> &A, const T v, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::pow(A.data[i], v);
    }

    template <typename T>
    inline void rpowscalar_cpu(const T v, const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::pow(v, A.data[i]);
    }

    template <typename T>
    inline void sqrt_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::sqrt(A.data[i]);
    }

    template <typename T>
    inline void log_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::log(A.data[i]);
    }

    template <typename T>
    inline void exp_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::exp(A.data[i]);
    }

    template <typename T>
    inline void sin_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::sin(A.data[i]);
    }

    template <typename T>
    inline void cos_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::cos(A.data[i]);
    }

    template <typename T>
    inline void tan_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = std::tan(A.data[i]);
    }

    template <typename T>
    inline void invert_cpu(const Tensor<T> &A, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = ~A.data[i];
    }

    template <>
    inline void invert_cpu<bool>(const Tensor<bool> &A, Tensor<bool> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
            C.data[i] = !A.data[i];
    }
}

namespace deepx::metal::kernels
{
#if defined(__APPLE__) && TARGET_OS_OSX && defined(__OBJC__)
    inline deepx::metal::common::MetalKernelRuntime &elementwise_runtime()
    {
        static deepx::metal::common::MetalKernelRuntime rt;
        return rt;
    }

    // ── ADD ──
    inline bool add_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("add_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool add_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("add_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool add_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("add_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool add_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("add_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool add_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("add_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── SUB ──
    inline bool sub_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("sub_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool sub_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("sub_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool sub_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("sub_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool sub_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("sub_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool sub_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("sub_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── MUL ──
    inline bool mul_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("mul_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool mul_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("mul_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool mul_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("mul_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool mul_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("mul_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool mul_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("mul_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── DIV ──
    inline bool div_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("div_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── MAX ──
    inline bool max_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("max_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool max_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("max_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool max_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("max_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool max_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("max_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool max_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("max_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── MIN ──
    inline bool min_f32(const float *a, const float *b, float *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("min_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool min_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("min_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool min_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("min_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool min_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("min_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool min_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n) {
        return elementwise_runtime().dispatch_binary_1d("min_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── RELU ──
    inline bool relu_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("relu_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool relu_i8(const int8_t *x, int8_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("relu_i8", x, y, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool relu_i16(const int16_t *x, int16_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("relu_i16", x, y, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool relu_i32(const int32_t *x, int32_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("relu_i32", x, y, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool relu_i64(const int64_t *x, int64_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("relu_i64", x, y, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── NEG ──
    inline bool neg_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("neg_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool neg_i8(const int8_t *x, int8_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("neg_i8", x, y, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool neg_i16(const int16_t *x, int16_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("neg_i16", x, y, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool neg_i32(const int32_t *x, int32_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("neg_i32", x, y, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool neg_i64(const int64_t *x, int64_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("neg_i64", x, y, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── ABS ──
    inline bool abs_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("abs_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }
    inline bool abs_i8(const int8_t *x, int8_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("abs_i8", x, y, static_cast<uint32_t>(n), sizeof(int8_t));
    }
    inline bool abs_i16(const int16_t *x, int16_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("abs_i16", x, y, static_cast<uint32_t>(n), sizeof(int16_t));
    }
    inline bool abs_i32(const int32_t *x, int32_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("abs_i32", x, y, static_cast<uint32_t>(n), sizeof(int32_t));
    }
    inline bool abs_i64(const int64_t *x, int64_t *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("abs_i64", x, y, static_cast<uint32_t>(n), sizeof(int64_t));
    }

    // ── SQRT ──
    inline bool sqrt_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("sqrt_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── EXP ──
    inline bool exp_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("exp_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── LOG ──
    inline bool log_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("log_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── SIN ──
    inline bool sin_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("sin_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── COS ──
    inline bool cos_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("cos_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

    // ── TAN ──
    inline bool tan_f32(const float *x, float *y, int64_t n) {
        return elementwise_runtime().dispatch_unary_1d("tan_f32", x, y, static_cast<uint32_t>(n), sizeof(float));
    }

#else
    // ── Stubs for non-ObjC / non-Apple platforms ──
    inline bool add_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool add_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool add_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool add_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool add_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }
    inline bool sub_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool sub_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool sub_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool sub_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool sub_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }
    inline bool mul_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool mul_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool mul_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool mul_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool mul_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }
    inline bool div_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool max_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool max_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool max_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool max_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool max_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }
    inline bool min_f32(const float *, const float *, float *, int64_t) { return false; }
    inline bool min_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool min_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool min_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool min_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }
    inline bool relu_f32(const float *, float *, int64_t) { return false; }
    inline bool relu_i8(const int8_t *, int8_t *, int64_t) { return false; }
    inline bool relu_i16(const int16_t *, int16_t *, int64_t) { return false; }
    inline bool relu_i32(const int32_t *, int32_t *, int64_t) { return false; }
    inline bool relu_i64(const int64_t *, int64_t *, int64_t) { return false; }
    inline bool neg_f32(const float *, float *, int64_t) { return false; }
    inline bool neg_i8(const int8_t *, int8_t *, int64_t) { return false; }
    inline bool neg_i16(const int16_t *, int16_t *, int64_t) { return false; }
    inline bool neg_i32(const int32_t *, int32_t *, int64_t) { return false; }
    inline bool neg_i64(const int64_t *, int64_t *, int64_t) { return false; }
    inline bool abs_f32(const float *, float *, int64_t) { return false; }
    inline bool abs_i8(const int8_t *, int8_t *, int64_t) { return false; }
    inline bool abs_i16(const int16_t *, int16_t *, int64_t) { return false; }
    inline bool abs_i32(const int32_t *, int32_t *, int64_t) { return false; }
    inline bool abs_i64(const int64_t *, int64_t *, int64_t) { return false; }
    inline bool sqrt_f32(const float *, float *, int64_t) { return false; }
    inline bool exp_f32(const float *, float *, int64_t) { return false; }
    inline bool log_f32(const float *, float *, int64_t) { return false; }
    inline bool sin_f32(const float *, float *, int64_t) { return false; }
    inline bool cos_f32(const float *, float *, int64_t) { return false; }
    inline bool tan_f32(const float *, float *, int64_t) { return false; }

#endif
} // namespace deepx::metal::kernels

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP
