#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_COMPARE_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_compare.cuh"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    // CUDA kernel函数声明

    template <typename T>
    struct maxDispatcher<miaobyte, T>
    {
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size)
            {
                throw TensorShapeError("max");
            }
            launch_max(A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct maxscalarDispatcher<miaobyte, T>
    {
        static void maxscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size)
            {
                throw TensorShapeError("maxscalar");
            }
            launch_maxscalar(A.data, scalar, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct minDispatcher<miaobyte, T>
    {
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size)
            {
                throw TensorShapeError("min");
            }

            launch_min(A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct minscalarDispatcher<miaobyte, T>
    {
        static void minscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size)
            {
                throw TensorShapeError("minscalar");
            }

            launch_minscalar(A.data, scalar, C.data, A.shape.size);
        }
    };
    // equal(A,B)=>C
    template <typename T,typename MaskT>
    struct equalDispatcher<miaobyte, T,MaskT>
    {
        static void equal(const Tensor<T> &A, const Tensor<T> &B, float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape.size != B.shape.size || A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("equal");
            }
            if (epsilon < 0)
            {
                throw std::invalid_argument("equal epsilon must be positive");
            }
            launch_equal(A.data, B.data, epsilon, mask.data, A.shape.size);
        }
    };
    // equalscalar(A,scalar)=>C
    template <typename T,typename MaskT>
    struct equalscalarDispatcher<miaobyte, T,MaskT>
    {
        static void equalscalar(const Tensor<T> &A, const T scalar, float epsilon, Tensor<MaskT> &mask)
        {
            if (A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("equalscalar");
            }
            if (epsilon < 0)
            {
                throw std::invalid_argument("equal epsilon must be positive");
            }
            launch_equalscalar(A.data, scalar, epsilon, mask.data, A.shape.size);
        }
    };

    // less(A,B)=>C
    template <typename T,typename MaskT>
    struct lessDispatcher<miaobyte, T,MaskT>
    {
        static void less(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape.size != B.shape.size || A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("less");
            }
            launch_less(A.data, B.data, mask.data, A.shape.size);
        }
    };
    // lessscalar(A,scalar)=>C
    template <typename T,typename MaskT>
    struct lessscalarDispatcher<miaobyte, T,MaskT>
    {
        static void lessscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("lessscalar");
            }
            launch_lessscalar(A.data, scalar, mask.data, A.shape.size);
        }
    };
    // greater(A,B)=>C
    template <typename T,typename MaskT>
    struct greaterDispatcher<miaobyte, T,MaskT>
    {
        static void greater(const Tensor<T> &A, const Tensor<T> &B, Tensor<MaskT> &mask)
        {
            if (A.shape.size != B.shape.size || A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("greater");
            }
            launch_greater(A.data, B.data, mask.data, A.shape.size);
        }
    };
    // greaterscalar(A,scalar)=>C
    template <typename T,typename MaskT>
    struct greaterscalarDispatcher<miaobyte, T,MaskT>
    {
        static void greaterscalar(const Tensor<T> &A, const T scalar, Tensor<MaskT> &mask)
        {
            if (A.shape.size != mask.shape.size)
            {
                throw TensorShapeError("greaterscalar");
            }
            launch_greaterscalar(A.data, scalar, mask.data, A.shape.size);
        }
    };
    // switch(tensors,cases)=>C
    template <typename T,typename casesT>
    struct switchDispatcher<miaobyte, T,casesT>
    {
        static void Switch(const vector<Tensor<T> *> tensors, const Tensor<casesT> &cases, Tensor<T> &C)
        {
            if (cases.shape.size != C.shape.size)
            {
                throw TensorShapeError("Switch");
            }
            
            vector<const T *> tensorsData(tensors.size());
            for (int i = 0; i < tensors.size(); i++)
            {
                tensorsData[i] = tensors[i]->data;
            }
            
            launch_switch(tensorsData.data(), tensors.size(), cases.data, C.data, C.shape.size);
        }
    };

}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
