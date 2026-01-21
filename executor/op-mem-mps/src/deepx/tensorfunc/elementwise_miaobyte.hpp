#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP

#include <stdexcept>
#include <type_traits>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_common.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            detail::assert_same_shape(A, B, C);

#if defined(__APPLE__) && TARGET_OS_OSX
            // Try Metal path for supported dtypes. Current tensors are host-backed,
            // so this does staging copies (correctness-first). If Metal is unavailable,
            // fall back to the CPU implementation below.
            bool ok = false;
            if constexpr (std::is_same_v<T, float>)
            {
                ok = deepx::mps::kernels::add_f32(A.data, B.data, C.data, A.shape.size);
            }
#if defined(__FLT16_MANT_DIG__)
            else if constexpr (std::is_same_v<T, _Float16>)
            {
                ok = deepx::mps::kernels::add_f16(A.data, B.data, C.data, A.shape.size);
            }
#endif
            else if constexpr (std::is_same_v<T, int8_t>)
            {
                ok = deepx::mps::kernels::add_i8(A.data, B.data, C.data, A.shape.size);
            }
            else if constexpr (std::is_same_v<T, int16_t>)
            {
                ok = deepx::mps::kernels::add_i16(A.data, B.data, C.data, A.shape.size);
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                ok = deepx::mps::kernels::add_i32(A.data, B.data, C.data, A.shape.size);
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                ok = deepx::mps::kernels::add_i64(A.data, B.data, C.data, A.shape.size);
            }

            if (ok)
            {
                return;
            }
#endif
            detail::add_cpu(A, B, C);
        }
    };
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAOBYTE_HPP
