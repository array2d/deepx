#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    template <typename Author, typename T>
    struct addDispatcher
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            throw NotImplementError("add");
        }
    };

    template <typename Author, typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        addDispatcher<Author, T>::add(A, B, C);
    }

    template <typename Author, typename T>
    struct addscalarDispatcher
    {
        static void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output){
            throw NotImplementError("addscalar");
        }
    };

    template <typename Author, typename T>
    void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        addscalarDispatcher<Author, T>::addscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct subDispatcher
    {
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C){
            throw NotImplementError("sub");
        }
    };

    template <typename Author, typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        subDispatcher<Author, T>::sub(A, B, C);
    }

    template <typename Author, typename T>
    struct subscalarDispatcher
    {
        static void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output){
            throw NotImplementError("subscalar");
        }
    };

    template <typename Author, typename T>
    void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        subscalarDispatcher<Author, T>::subscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct mulDispatcher
    {
        static void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        mulDispatcher<Author, T>::mul(A, B, C);
    }

    template <typename Author, typename T>
    struct mulscalarDispatcher
    {
        static void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        mulscalarDispatcher<Author, T>::mulscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct muladdDispatcher
    {
        static void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;
    };

    template <typename Author, typename T>
    void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
    {
        muladdDispatcher<Author, T>::muladd(A, B, C, D);
    }

    template <typename Author, typename T>
    struct muladdscalarDispatcher
    {
        static void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;
    };

    template <typename Author, typename T>
    void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
    {
        muladdscalarDispatcher<Author, T>::muladdscalar(A, B, alpha, C, beta, D);
    }

    template <typename Author, typename T>
    struct mulscalaraddDispatcher
    {
        static void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
    {
        mulscalaraddDispatcher<Author, T>::mulscalaradd(A, alpha, B, beta, C);
    }

    template <typename Author, typename T>
    struct divDispatcher
    {
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        divDispatcher<Author, T>::div(A, B, C);
    }

    template <typename Author, typename T>
    struct divscalarDispatcher
    {
        static void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        divscalarDispatcher<Author, T>::divscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct rdivscalarDispatcher
    {
        static void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        rdivscalarDispatcher<Author, T>::rdivscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct divaddDispatcher
    {
        static void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;
    };

    template <typename Author, typename T>
    void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
    {
        divaddDispatcher<Author, T>::divadd(A, B, C, D);
    }

    template <typename Author, typename T>
    struct divscalaraddDispatcher
    {
        static void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
    {
        divscalaraddDispatcher<Author, T>::divscalaradd(A, alpha, B, beta, C);
    }

    template <typename Author, typename T>
    struct divaddbetaDispatcher
    {
        static void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;
    };

    template <typename Author, typename T>
    void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
    {
        divaddbetaDispatcher<Author, T>::divaddbeta(A, B, alpha, C, beta, D);
    }

    template <typename Author, typename T>
    struct sqrtDispatcher
    {
        static void sqrt(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void sqrt(const Tensor<T> &input, Tensor<T> &output)
    {
        sqrtDispatcher<Author, T>::sqrt(input, output);
    }

    template <typename Author, typename T>
    struct powDispatcher
    {
        static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        powDispatcher<Author, T>::pow(A, B, C);
    }

    template <typename Author, typename T>
    struct powscalarDispatcher
    {
        static void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        powscalarDispatcher<Author, T>::powscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct logDispatcher
    {
        static void log(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void log(const Tensor<T> &input, Tensor<T> &output)
    {
        logDispatcher<Author, T>::log(input, output);
    }

    template <typename Author, typename T>
    struct expDispatcher
    {
        static void exp(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void exp(const Tensor<T> &input, Tensor<T> &output)
    {
        expDispatcher<Author, T>::exp(input, output);
    }

    template <typename Author, typename T>
    struct sinDispatcher
    {
        static void sin(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void sin(const Tensor<T> &input, Tensor<T> &output)
    {
        sinDispatcher<Author, T>::sin(input, output);
    }

    template <typename Author, typename T>
    struct cosDispatcher
    {
        static void cos(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void cos(const Tensor<T> &input, Tensor<T> &output)
    {
        cosDispatcher<Author, T>::cos(input, output);
    }

    template <typename Author, typename T>
    struct tanDispatcher
    {
        static void tan(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void tan(const Tensor<T> &input, Tensor<T> &output)
    {
        tanDispatcher<Author, T>::tan(input, output);
    }

    template <typename Author, typename T>
    struct maxDispatcher
    {
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        maxDispatcher<Author, T>::max(A, B, C);
    }

    template <typename Author, typename T>
    struct maxgradDispatcher
    {
        static void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;
    };

    template <typename Author, typename T>
    void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
    {
        maxgradDispatcher<Author, T>::maxgrad(A, B, A_grad, B_grad, output_grad);
    }

    template <typename Author, typename T>
    struct maxscalarDispatcher
    {
        static void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        maxscalarDispatcher<Author, T>::maxscalar(A, b, C);
    }

    template <typename Author, typename T>
    struct maxscalargradDispatcher
    {
        static void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };

    template <typename Author, typename T>
    void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
    {
        maxscalargradDispatcher<Author, T>::maxscalargrad(A, b, A_grad, output_grad);
    }

    template <typename Author, typename T>
    struct minDispatcher
    {
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        minDispatcher<Author, T>::min(A, B, C);
    }

    template <typename Author, typename T>
    struct mingradDispatcher
    {
        static void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;
    };

    template <typename Author, typename T>
    void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
    {
        mingradDispatcher<Author, T>::mingrad(A, B, A_grad, B_grad, output_grad);
    }

    template <typename Author, typename T>
    struct minscalarDispatcher
    {
        static void minscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        minscalarDispatcher<Author, T>::minscalar(A, b, C);
    }

    template <typename Author, typename T>
    struct minscalargradDispatcher
    {
        static void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };

    template <typename Author, typename T>
    void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
    {
        minscalargradDispatcher<Author, T>::minscalargrad(A, b, A_grad, output_grad);
    }
} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
