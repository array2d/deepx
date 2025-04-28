#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_basic.cuh"

#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{   
    //todtype
    template <typename T,typename Dtype>
    void todtype(const Tensor<T> &input, Tensor<Dtype> &output){
        if (input.shape.size != output.shape.size || input.shape.size != output.shape.size) {
            throw TensorShapeError("todtype");
        }
        launch_todtype(input.data, output.data, input.shape.size);
    };
 
    //add
    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) {
                throw TensorShapeError("add");
            }
            launch_add(A.data, B.data, C.data, A.shape.size);
           
        }   
    };

    template <typename T>
    struct addscalarDispatcher<miaobyte, T>
    {
        static void addscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) {
                throw TensorShapeError("addscalar");
            }
            launch_addscalar(A.data, scalar, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct subDispatcher<miaobyte, T>
    {
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) { 
                throw TensorShapeError("sub");
            }
            launch_sub(A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct subscalarDispatcher<miaobyte, T>
    {
        static void subscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("subscalar");
            }
            launch_subscalar(A.data, scalar, C.data, A.shape.size);
        }
    };  

    template <typename T>
    struct mulDispatcher<miaobyte, T>
    {
        static void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) { 
                throw TensorShapeError("mul");
            }
            launch_mul(A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct mulscalarDispatcher<miaobyte, T>
    {
        static void mulscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("mulscalar");    
            }
            launch_mulscalar(A.data, scalar, C.data, A.shape.size);
        }
    };  

    template <typename T>
    struct divDispatcher<miaobyte, T>
    {
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)   
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) { 
                throw TensorShapeError("div");
            }
            launch_div(A.data, B.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct divscalarDispatcher<miaobyte, T>
    {
        static void divscalar(const Tensor<T> &A, const T scalar, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("divscalar");
            }
            launch_divscalar(A.data, scalar, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct rdivscalarDispatcher<miaobyte, T>
    {
        static void rdivscalar(const T scalar, const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("rdivscalar");
            }
            launch_rdivscalar(scalar, A.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct invertDispatcher<miaobyte, T>
    {
        static void invert(const Tensor<T> &A, Tensor<T> &C)
        {
            if (A.shape.size != C.shape.size) { 
                throw TensorShapeError("invert");
            }
            launch_invert( A.data, C.data, A.shape.size);
        }
    };

    template <typename T>
    struct dropoutDispatcher<miaobyte, T>
    {
        static void dropout(const Tensor<T> &A, const float p,const unsigned int seed, Tensor<T> &C)
        {
            launch_dropout(A.data, p, seed, C.data, A.shape.size);
        }           
    };
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
