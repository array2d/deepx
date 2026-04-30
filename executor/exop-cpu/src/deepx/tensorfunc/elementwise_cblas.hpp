#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP

#include "cblas.h"

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tensorfunc
{

    // double特化
    template <>
    struct addDispatcher<cblas, double>
    {
        static void add(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                if (std::addressof(A) != std::addressof(C))
                {
                    cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
                }
                cblas_daxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
    // float特化
    template <>
    struct addDispatcher<cblas, float>
    {
        static void add(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                if (std::addressof(A) != std::addressof(C))
                {
                    cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
                }
                cblas_saxpy(B.shape.size, 1.0f, B.data, 1, C.data, 1);
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };

    // double特化
    template <>
    struct subDispatcher<cblas, double>
    {
        static void sub(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
        {
            if (A.shape == B.shape && A.shape == C.shape)
            {
                if (std::addressof(A) != std::addressof(C))
                {
                    cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
                }
                cblas_daxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
    // float特化
    template <>
    struct subDispatcher<cblas, float>
    {
        static void sub(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
        {
            // 先复制A到C，再累加B (C = 1*A - 1*B)
            if (A.shape == B.shape && A.shape == C.shape)
            {
                if (std::addressof(A) != std::addressof(C))
                {
                    cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
                }
                cblas_saxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
            }
            else
            {
                throw std::invalid_argument("shape mismatch");
            }
        }
    };
} // namespace deepx::tensorfunc
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP
