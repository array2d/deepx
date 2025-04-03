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


    // A+B=>C
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

    // A+scalar=>C
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

    // A-B=>C
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

    // A-scalar=>C
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

    // A*B=>C
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

    // A*scalar=>C
    template <typename Author, typename T>
    void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        mulscalarDispatcher<Author, T>::mulscalar(input, value, output);
    }

 
  
    template <typename Author, typename T>
    struct divDispatcher
    {
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // A/B=>C
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

    // A/scalar=>C
    template <typename Author, typename T>
    void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        divscalarDispatcher<Author, T>::divscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct rdivscalarDispatcher
    {
        static void rdivscalar(const T value, const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    // scalar/A=>C
    template <typename Author, typename T>
    void rdivscalar(const T value, const Tensor<T> &input, Tensor<T> &output)
    {
        rdivscalarDispatcher<Author, T>::rdivscalar(value, input, output);
    }

    
    template <typename Author, typename T,typename = void>
    struct sqrtDispatcher
    {
        static void sqrt(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    // sqrt(A)=>C   
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

    // A^B=>C
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

    // A^scalar=>C
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

    // log(A)=>C
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

    // exp(A)=>C
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

    // sin(A)=>C
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

    // cos(A)=>C
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

    // tan(A)=>C
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

    // max(A,B)=>C
    template <typename Author, typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        maxDispatcher<Author, T>::max(A, B, C);
    }

    

    template <typename Author, typename T>
    struct maxscalarDispatcher
    {
        static void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };

    // max(A,scalar)=>C
    template <typename Author, typename T>
    void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        maxscalarDispatcher<Author, T>::maxscalar(A, b, C);
    }

 

    template <typename Author, typename T>
    struct minDispatcher
    {
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // min(A,B)=>C
    template <typename Author, typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        minDispatcher<Author, T>::min(A, B, C);
    }

    template <typename Author, typename T>
    struct minscalarDispatcher
    {
        static void minscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };

    // min(A,scalar)=>C
    template <typename Author, typename T>
    void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        minscalarDispatcher<Author, T>::minscalar(A, b, C);
    }
    
    template <typename Author, typename T>
    struct compareDispatcher
    {
        static void compare(const Tensor<T> &A, const Tensor<T> &B, Tensor<float> &mask) = delete;
    };

    // compare(A,B)=>mask
    // if A[i]==B[i], mask[i]=0.5
    // if A[i]>B[i], mask[i]=0
    // if A[i]<B[i], mask[i]=1
    template <typename Author, typename T>
    void compare(const Tensor<T> &A, const Tensor<T> &B,Tensor<float> &mask)
    {
        compareDispatcher<Author, T>::compare(A, B, mask);
    }

    template <typename Author, typename T>
    struct comparescalarDispatcher
    {
        static void comparescalar(const Tensor<T> &A, const T scalar, Tensor<float> &mask) = delete;
    };

    template <typename Author, typename T>
    void comparescalar(const Tensor<T> &A, const T scalar, Tensor<float> &mask)
    {
        comparescalarDispatcher<Author, T>::comparescalar(A, scalar, mask);
    }
} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
