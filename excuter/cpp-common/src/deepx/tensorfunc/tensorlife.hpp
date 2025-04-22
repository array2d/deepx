#ifndef DEEPX_TENSORFUNC_TENSORLIFE_HPP
#define DEEPX_TENSORFUNC_TENSORLIFE_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    //New
    template < typename T>
    Tensor<T> New(const std::vector<int> &shape);

    template <typename T>
    Tensor<T> New(const std::initializer_list<int> &shape){
        std::vector<int> shape_vec(shape);
        return New<T>(shape_vec);
    }
 
    //copy
    template <typename T>
    void copy(const Tensor<T> &src,Tensor<T> &dst);

    //rename
    //通过tf直接实现
}
#endif