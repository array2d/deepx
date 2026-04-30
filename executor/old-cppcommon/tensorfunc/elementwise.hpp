#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    //todtype
    template <typename T,typename Dtype>
    void todtype(const Tensor<T> &input, Tensor<Dtype> &output);
 
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
        static void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
        {
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
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            throw NotImplementError("sub");
        }
    };

    // A-B=>C
    template <typename Author, typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        subDispatcher<Author, T>::sub(A, B, C);
    }

    // A-scalar=>C
    template <typename Author, typename T>
    struct subscalarDispatcher
    {
        static void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
        {
            throw NotImplementError("subscalar");
        }
    };
    template <typename Author, typename T>
    void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        subscalarDispatcher<Author, T>::subscalar(input, value, output);
    }



    //scalar-A=>C
    template <typename Author, typename T>
    struct rsubscalarDispatcher
    {
        static void rsubscalar(const T value, const Tensor<T> &input, Tensor<T> &output) = delete;
    };
    template <typename Author, typename T>
    void rsubscalar(const T value, const Tensor<T> &input, Tensor<T> &output)
    {
        rsubscalarDispatcher<Author, T>::rsubscalar(value, input, output);
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

    // A^B=>C
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

    // A^scalar=>C
    template <typename Author, typename T>
    void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        powscalarDispatcher<Author, T>::powscalar(input, value, output);
    }

    template <typename Author, typename T>
    struct rpowscalarDispatcher
    {
        static void rpowscalar(const T value, const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    // scalar^A=>C
    template <typename Author, typename T>
    void rpowscalar(const T value, const Tensor<T> &input, Tensor<T> &output)
    {
        rpowscalarDispatcher<Author, T>::rpowscalar(value, input, output);
    }

    template <typename Author, typename T, typename = void>
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

    // equal(A,B)=>mask

    template <typename Author, typename T, typename MaskT>
    struct equalDispatcher
    {
        static void equal(const Tensor<T> &A, const Tensor<T> &B, float epsilon, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void equal(const Tensor<T> &A, const Tensor<T> &B, float epsilon, Tensor<MaskT> &mask)
    {
        equalDispatcher<Author, T, MaskT>::equal(A, B, epsilon, mask);
    }

    // equal(A,scalar)=>mask
    template <typename Author, typename T, typename MaskT>
    struct equalscalarDispatcher
    {
        static void equalscalar(const Tensor<T> &A, const T scalar, float epsilon, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void equalscalar(const Tensor<T> &A, const T scalar, float epsilon, Tensor<MaskT> &mask)
    {
        equalscalarDispatcher<Author, T, MaskT>::equalscalar(A, scalar, epsilon, mask);
    }
    //notequal(A,B)=>mask
    template <typename Author, typename T, typename MaskT>
    struct notequalDispatcher
    {
        static void notequal(const Tensor<T> &A, const Tensor<T> &B,const float epsilon, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void notequal(const Tensor<T> &A, const Tensor<T> &B,const float epsilon, Tensor<MaskT> &mask)
    {
        notequalDispatcher<Author, T, MaskT>::notequal(A, B, epsilon, mask);
    }

    // notequal(A,scalar)=>mask
    template <typename Author, typename T, typename MaskT>
    struct notequalscalarDispatcher
    {
        static void notequalscalar(const Tensor<T> &A, const T scalar,const float epsilon, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void notequalscalar(const Tensor<T> &A, const T scalar,const float epsilon, Tensor<MaskT> &mask)
    {
        notequalscalarDispatcher<Author, T, MaskT>::notequalscalar(A, scalar, epsilon, mask);
    }

    // less(A,B)=>mask
    template <typename Author, typename T, typename MaskT>
    struct lessDispatcher
    {
        static void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
    {
        lessDispatcher<Author, T, MaskT>::less(A, B, mask);
    }

    // less(A,scalar)=>mask
    template <typename Author, typename T, typename MaskT>
    struct lessscalarDispatcher
    {
        static void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
    {
        lessscalarDispatcher<Author, T, MaskT>::lessscalar(A, scalar, mask);
    }

    // greater(A,B)=>C
    template <typename Author, typename T, typename MaskT>
    struct greaterDispatcher
    {
        static void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
    {
        greaterDispatcher<Author, T, MaskT>::greater(A, B, mask);
    }

    // greater(A,scalar)=>C
    template <typename Author, typename T, typename MaskT>
    struct greaterscalarDispatcher
    {
        static void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask) = delete;
    };

    template <typename Author, typename T, typename MaskT>
    void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
    {
        greaterscalarDispatcher<Author, T, MaskT>::greaterscalar(A, scalar, mask);
    }

    // switch(tensors,cases)=>C
    template <typename Author, typename T, typename casesT>
    struct switchDispatcher
    {
        static void Switch(const vector<Tensor<T> *> tensors, const Tensor<casesT> &cases, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T, typename casesT>
    void Switch(const vector<Tensor<T> *> tensors, const Tensor<casesT> &cases, Tensor<T> &C)
    {
        switchDispatcher<Author, T, casesT>::Switch(tensors, cases, C);
    }

    // invert(A)=>C
    template <typename Author, typename T>
    struct invertDispatcher
    {
        static void invert(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void invert(const Tensor<T> &input, Tensor<T> &output)
    {
        invertDispatcher<Author, T>::invert(input, output);
    };
    
} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
