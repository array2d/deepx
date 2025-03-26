#ifndef DEEPX_TENSORFUNC_MATMUL_HPP
#define DEEPX_TENSORFUNC_MATMUL_HPP

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "stdutil/error.hpp"
namespace deepx::tensorfunc
{
    bool check_matmul_shape(const Shape &a, const Shape &b)
    {
        if (a[-1] != b[-2])
        {
            return false;
        }
        if (a.dim != b.dim)
        {
            return false;
        }
        for (int i = 0; i < a.dim - 2; ++i)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    template <typename Author, typename T>
    struct matmulDispatcher
    {
        static void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            throw NotImplementError("matmul");
        }
    };

    template <typename Author, typename T>
    void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        matmulDispatcher<Author, T>::matmul(A, B, C);
    }

    template <typename Author, typename T>
    struct matmuladdDispatcher
    {
        static void matmuladd(const Tensor<T> &A, const Tensor<T> &B, const T &alpha, const T &beta, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void matmuladd(const Tensor<T> &A, const Tensor<T> &B, const T &alpha, const T &beta, Tensor<T> &C)
    {
        matmuladdDispatcher<Author, T>::matmuladd(A, B, alpha, beta, C);
    }

}

#endif
