#ifndef DEEPX_TENSORFUNC_REDUCE_HPP
#define DEEPX_TENSORFUNC_REDUCE_HPP

 #include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    template <typename Author, typename T>
    struct reducesumDispatcher
    {
        static void reducesum(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B) = delete;
    };
    template <typename Author, typename T>
    void reducesum(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B)
    {
        reducesumDispatcher<Author, T>::reducesum(A, axis, keepdims, B);
    }
    
    template <typename Author, typename T>
    struct reduceprodDispatcher
    {
        static void reduceprod(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B) = delete;
    };
    
    template <typename Author, typename T>
    void reduceprod(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B)
    {
        reduceprodDispatcher<Author, T>::reduceprod(A, axis, keepdims, B);
    }

    template <typename Author, typename T>
    struct reducemaxDispatcher
    {
        static void reducemax(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B) = delete;
    };
    template <typename Author, typename T>
    void reducemax(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B)
    {
        reducemaxDispatcher<Author, T>::reducemax(A, axis, keepdims, B);
    }
    
    template <typename Author, typename T>
    struct reduceminDispatcher
    {
        static void reducemin(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B) = delete;
    };
    template <typename Author, typename T>
    void reducemin(const Tensor<T> &A, const int axis,const bool keepdims, Tensor<T> &B)
    {
        reduceminDispatcher<Author, T>::reducemin(A, axis, keepdims, B);
    }
 
}
#endif // DEEPX_TENSORFUNC_REDUCE_HPP
