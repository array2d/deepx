#ifndef DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"
#include "stdutil/print.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/io.hpp"

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
}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP