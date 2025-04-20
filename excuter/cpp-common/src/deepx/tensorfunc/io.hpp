#ifndef DEEPX_TENSORFUNC_IO_HPP
#define DEEPX_TENSORFUNC_IO_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc{
    
    template <typename Author,typename T>
    struct printDispatcher{
        static void print(const Tensor<T> &t, const std::string &f="")=delete;
    };

    template <typename Author, typename T>
    void print(const Tensor<T> &t, const std::string &f=""){
        printDispatcher<Author,T>::print(t, f);
    }

    template <typename T>
    void save(Tensor<T> &tensor,const std::string &path);

    template <typename T>
    pair<std::string,shared_ptr<Tensor<T>>> load(const std::string &path);

    pair<std::string,Shape> loadShape(const std::string &path);
    
}

#endif // DEEPX_TENSORFUNC_IO_HPP
