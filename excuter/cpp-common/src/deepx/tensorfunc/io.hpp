#ifndef DEEPX_TENSORFUNC_IO_HPP
#define DEEPX_TENSORFUNC_IO_HPP

#include "deepx/tensor.hpp"
#include "stdutil/fs.hpp"

namespace deepx::tensorfunc{
    
    template <typename Author,typename T>
    struct printDispatcher{
        static void print(const Tensor<T> &t, const std::string &f="")=delete;
    };

    template <typename Author, typename T>
    void print(const Tensor<T> &t, const std::string &f=""){
        printDispatcher<Author,T>::print(t, f);
    }
    
 
   
 
}

#endif // DEEPX_TENSORFUNC_IO_HPP
