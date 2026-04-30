#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"
 
namespace deepx::tensorfunc
{   
    //constant
    template <typename Author, typename T>
    struct constantDispatcher
    {
        static void constant(Tensor<T> &tensor, const T value ) = delete;
    };

    template <typename Author, typename T>
    void constant(Tensor<T> &tensor, const T value = T(0))
    {
        constantDispatcher<Author, T>::constant(tensor, value);
    }

    //dropout(A,p)=>C
    template <typename Author, typename T>
    struct dropoutDispatcher
    {
        static void dropout(Tensor<T> &input, const float p,const unsigned int seed) = delete;
    };

    template <typename Author, typename T>
    void dropout(Tensor<T> &input, const float p,const unsigned int seed)
    {
        dropoutDispatcher<Author, T>::dropout(input, p, seed);
    }
    

    //arange
    template <typename Author, typename T>
    struct arangeDispatcher
    {
        static void arange(Tensor<T> &tensor, const T start , const T step  ) = delete;
    };

    template <typename Author, typename T>
    void arange(Tensor<T> &tensor, const T start = T(0), const T step = T(1))
    {
        arangeDispatcher<Author, T>::arange(tensor, start, step);
    }

    //uniform
    template <typename Author, typename T>
    struct uniformDispatcher
    {
        static void uniform(Tensor<T> &tensor, const T low  , const T high  , const unsigned int seed) = delete;
    }; 

    template <typename Author, typename T>
    void uniform(Tensor<T> &tensor, const T low = T(0), const T high = T(1), const unsigned int seed = 0)
    {
        uniformDispatcher<Author, T>::uniform(tensor, low, high, seed);
    }

    //normal
    template <typename Author, typename T>
    struct normalDispatcher
    {
        static void normal(Tensor<T> &tensor, const T mean  , const T stddev  , const unsigned int seed) = delete;
    };

    template <typename Author, typename T>
    void normal(Tensor<T> &tensor, const T mean = T(0), const T stddev = T(1), const unsigned int seed = 0)
    {
        normalDispatcher<Author, T>::normal(tensor, mean, stddev, seed);
    }
 
}
#endif
