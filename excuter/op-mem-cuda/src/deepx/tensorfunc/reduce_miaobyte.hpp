#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include <deepx/vector_combination.hpp>

#include "deepx/tensorfunc/reduce.hpp"

namespace deepx::tensorfunc
{
    
    template < typename T>
    struct reducemaxDispatcher<miaobyte, T>
    {
        static void reducemax(const Tensor<T> &A, const std::vector<int> &dims, Tensor<T> &B,const bool keepdims) {
            if (axis < 0) {
                axis += A.shape.dim;
            }
            if (axis >= A.shape.dim) {
                throw std::invalid_argument("Invalid axis for reducemax");
            }
            
        }
    };


    template < typename T>
    struct reduceminDispatcher<miaobyte, T>
    {
        static void reducemin(const Tensor<T> &A, const std::vector<int> &dims, Tensor<T> &B,const bool keepdims) {
            if (axis < 0) {
                axis += A.shape.dim;
            }
            if (axis >= A.shape.dim) {
                throw std::invalid_argument("Invalid axis for reducemin");
            }
            
        }
    };


    template <typename T>
    struct sumDispatcher<miaobyte, T>
    {
        static void sum(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result,const bool keepdims)
        {
             
            
        }
    };
     

    template <typename T>
    struct prodDispatcher<miaobyte, T>
    {
        static void prod(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result,const bool keepdims)
        {

        }
    };
}
#endif DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_HPP