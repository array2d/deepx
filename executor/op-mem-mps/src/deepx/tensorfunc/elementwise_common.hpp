#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP

#if defined(__APPLE__)
  #include <TargetConditionals.h>
#endif

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/mps_common.hpp"

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
    inline void add_cpu(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        for (int64_t i = 0; i < A.shape.size; ++i)
        {
            C.data[i] = A.data[i] + B.data[i];
        }
    }
}

namespace deepx::mps::kernels
{
#if defined(__APPLE__) && TARGET_OS_OSX && defined(__OBJC__)
    inline deepx::mps::common::MetalKernelRuntime &elementwise_runtime()
    {
        static deepx::mps::common::MetalKernelRuntime rt("elementwise_miaobyte.metal");
        return rt;
    }

    inline bool add_f32(const float *a, const float *b, float *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_f32", a, b, c, static_cast<uint32_t>(n), sizeof(float));
    }

#if defined(__FLT16_MANT_DIG__)
    inline bool add_f16(const _Float16 *a, const _Float16 *b, _Float16 *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_f16", a, b, c, static_cast<uint32_t>(n), 2);
    }
#endif

    inline bool add_i8(const int8_t *a, const int8_t *b, int8_t *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_i8", a, b, c, static_cast<uint32_t>(n), sizeof(int8_t));
    }

    inline bool add_i16(const int16_t *a, const int16_t *b, int16_t *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_i16", a, b, c, static_cast<uint32_t>(n), sizeof(int16_t));
    }

    inline bool add_i32(const int32_t *a, const int32_t *b, int32_t *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_i32", a, b, c, static_cast<uint32_t>(n), sizeof(int32_t));
    }

    inline bool add_i64(const int64_t *a, const int64_t *b, int64_t *c, int64_t n)
    {
        return elementwise_runtime().dispatch_binary_1d("add_i64", a, b, c, static_cast<uint32_t>(n), sizeof(int64_t));
    }

#else

    inline bool add_f32(const float *, const float *, float *, int64_t) { return false; }

#if defined(__FLT16_MANT_DIG__)
    inline bool add_f16(const _Float16 *, const _Float16 *, _Float16 *, int64_t) { return false; }
#endif

    inline bool add_i8(const int8_t *, const int8_t *, int8_t *, int64_t) { return false; }
    inline bool add_i16(const int16_t *, const int16_t *, int16_t *, int64_t) { return false; }
    inline bool add_i32(const int32_t *, const int32_t *, int32_t *, int64_t) { return false; }
    inline bool add_i64(const int64_t *, const int64_t *, int64_t *, int64_t) { return false; }

#endif
} // namespace deepx::mps::kernels

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_COMMON_HPP
