#ifndef DEEPX_TENSORFUNC_REDUCE_HPP
#define DEEPX_TENSORFUNC_REDUCE_HPP

 #include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{


    template <typename Author, typename T>
    struct reducemaxDispatcher
    {
        static void reducemax(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false) = delete;
    };
    template <typename Author, typename T>
    void reducemax(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false)
    {
        reducemaxDispatcher<Author, T>::reducemax(A, dims, B, keepdims);
    }
    
    template <typename Author, typename T>
    struct reduceminDispatcher
    {
        static void reducemin(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false) = delete;
    };
    template <typename Author, typename T>
    void reducemin(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false)
    {
        reduceminDispatcher<Author, T>::reducemin(A, dims, B, keepdims);
    }
    
    template <typename Author, typename T>
    struct  sumDispatcher
    {
        static void reducesum(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false) = delete;
    };
    template <typename Author, typename T>
    void sum(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false)
    {
        sumDispatcher<Author, T>::sum(A, dims, B, keepdims);
    }
    
    template <typename Author, typename T>
    struct  prodDispatcher
    {
        static void prod(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false) = delete;
    };
    
    template <typename Author, typename T>
    void prod(const Tensor<T> &A, const std::vector<int> &dims,Tensor<T> &B,const bool keepdims=false)
    {
        prodDispatcher<Author, T>::prod(A, dims, B, keepdims);
    }
}
#endif // DEEPX_TENSORFUNC_REDUCE_HPP
