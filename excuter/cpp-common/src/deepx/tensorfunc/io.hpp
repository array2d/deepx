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

    template <typename T>
    void save(Tensor<T> &tensor,const std::string &path);

    template <typename T>
    pair<std::string,shared_ptr<Tensor<T>>> load(const std::string &path);

    inline pair<std::string,Shape> loadShape(const std::string &path)
    {
        std::string shapepath = path + ".shape";
        std::ifstream shape_fs(shapepath, std::ios::binary);
        std::string shapedata((std::istreambuf_iterator<char>(shape_fs)), std::istreambuf_iterator<char>());
        Shape shape;
        shape.fromYaml(shapedata);
        std::string filename = stdutil::filename(path);
        std::string tensor_name = filename.substr(0, filename.find_last_of('.'));
        return std::make_pair(tensor_name, shape);
    }
    
}

#endif // DEEPX_TENSORFUNC_IO_HPP
