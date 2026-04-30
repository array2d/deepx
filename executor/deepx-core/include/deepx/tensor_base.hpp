#ifndef DEEPX_TENSORBASE_HPP
#define DEEPX_TENSORBASE_HPP

#include "deepx/shape.hpp"

namespace deepx
{
 
    struct TensorBase
    {
        Shape shape;
        TensorBase() = default;
        // 拷贝构造函数
        TensorBase(const TensorBase &other)
        {
            shape = other.shape;
        }

        // 移动构造函数
        TensorBase(TensorBase &&other) noexcept
        {
            shape = std::move(other.shape);
        }

        // 拷贝赋值运算符
        TensorBase &operator=(const TensorBase &other)
        {
            if (this != &other)
            {
                shape = other.shape;
            }
            return *this;
        }

        // 移动赋值运算符
        TensorBase &operator=(TensorBase &&other) noexcept
        {
            if (this != &other)
            {
                shape = std::move(other.shape);
            }
            return *this;
        }
    };

}
#endif
