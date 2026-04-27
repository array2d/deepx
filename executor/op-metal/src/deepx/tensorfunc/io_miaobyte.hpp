#ifndef DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"
#include "stdutil/print.hpp"
#include "stdutil/fs.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/io.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
namespace deepx::tensorfunc
{
    // 通用模板特化
    template <typename T>
    struct printDispatcher<miaobyte, T>
    {
        static void print(const Tensor<T> &t, const std::string &f = "")
        {
            Tensor<void> vt;
            vt.data = t.data;
            vt.shape = t.shape;
            vt.deleter = nullptr;
            stdutil::print(t.shape.shape, t.data, t.shape.dtype, f);
        }
    };

    // void类型的完全特化
    template <>
    struct printDispatcher<miaobyte, void>
    {
        static void print(const Tensor<void> &t, const std::string &f = "")
        {
            stdutil::print(t.shape.shape, t.data, t.shape.dtype, f);
        }
    };


    //load
    template <typename T>
    pair<std::string,shared_ptr<Tensor<T>>> load(const std::string &path)
    {
        // 加载shape
        pair<std::string,Shape> shape_name=Shape::loadShape(path);
        Shape shape=shape_name.second;
        std::string tensor_name=shape_name.first;
 

        // 检查T 和 shape.dtype 是否匹配
        if (shape.dtype != precision<T>())
        {
            throw std::runtime_error("调用load<" + precision_str(shape.dtype) + "> 不匹配: 需要 " + precision_str(shape.dtype) +
                                     " 类型，但文件为" + precision_str(precision<T>()) + " 类型");
        }
 
        shared_ptr<Tensor<T>> tensor = make_shared<Tensor<T>>(New<T>(shape.shape));
        tensor->loader(path+".data",tensor->data,tensor->shape.size);
        return std::make_pair(tensor_name, tensor);
    };

}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP