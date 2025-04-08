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

    template <typename Author, typename T>
    struct saveDispatcher{
        static void save(Tensor<T> &tensor,const std::string &path,int filebegin=0,int fileend=-1)=delete;
    };

    template <typename Author, typename T>
    void save(Tensor<T> &tensor,const std::string &path,int filebegin=0,int fileend=-1){
        saveDispatcher<Author,T>::save(tensor, path, filebegin, fileend);
    }

    template <typename Author, typename T>
    struct loadDispatcher{
        static Tensor<T> load(const std::string &path,int filebegin=0,int fileend=-1)=delete;
    };

    template <typename Author, typename T>
    Tensor<T> load(const std::string &path,int filebegin=0,int fileend=-1){
        return loadDispatcher<Author,T>::load(path, filebegin, fileend);
    }
}

#endif // DEEPX_TENSORFUNC_IO_HPP
