#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <memory>

#include "deepx/shape.hpp"
#include "deepx/dtype.hpp"
#include "deepx/tensorbase.hpp"

namespace deepx
{
    using namespace std;

    template <typename T>
    struct Tensor : public TensorBase
    {
        T *data;

        using NewFn = T *(*)(int);
        NewFn newer; //  申请内存

        using DeleteFn = void (*)(T *);
        DeleteFn deleter; // 释放内存

        using CopyFn = void (*)(T *, T *, int);
        CopyFn copyer; // 拷贝内存

        Tensor() = default;
        Tensor(const vector<int> &s)
        {
            shape = Shape(s);
        }
        Tensor(const Shape &s)
        {
            shape = s;
        }

        ~Tensor()
        {
            if (data && deleter)
            {
                deleter(data);
                data = nullptr;
            }
        }
        /**
         * 拷贝构造
         * 该构造函数用于创建一个新的Tensor对象，并将现有Tensor对象的内容复制到新对象中。
         * 它会分配新的内存并使用copyer函数将数据从源Tensor复制到新Tensor。
         */
        Tensor(const Tensor<T> &tensor)
        {
            shape = tensor.shape;
            device = tensor.device;
            newer = tensor.newer;
            deleter = tensor.deleter;
            copyer = tensor.copyer;

            data = newer(shape.size);
            copyer(tensor.data, data, tensor.shape.size);
        }

        /**
         * 移动构造
         * 该构造函数用于通过转移资源来创建一个新的Tensor对象。
         * 它会将源Tensor的资源（如数据指针）转移到新对象中，并将源Tensor的数据指针置为nullptr。
         * 这样可以避免不必要的内存分配，提高性能。
         */

        Tensor(Tensor<T> &&other) noexcept
        {
            shape = std::move(other.shape);
            device = other.device;

            deleter = other.deleter;
            copyer = other.copyer;
            newer = other.newer;

            data = other.data;

            other.data = nullptr;

            other.deleter = nullptr;
            other.copyer = nullptr;
            other.newer = nullptr;
        }

        /**
         * 拷贝赋值运算符
         * 该运算符用于将一个Tensor对象的内容赋值给另一个Tensor对象。
         * 它会先检查自赋值的情况，然后使用copyer函数将数据从源Tensor复制到目标Tensor。
         * 需要注意的是，目标Tensor的原有数据会被释放。
         */

        Tensor<T> &operator=(const Tensor<T> &tensor)
        {
            if (this == &tensor)
                return *this;

            shape = tensor.shape;
            device = tensor.device;
            deleter = tensor.deleter;
            copyer = tensor.copyer;
            newer = tensor.newer;

            data = newer(shape.size);
            if (data != nullptr)
            {
                deleter(data);
            }
            copyer(tensor.data, data, tensor.shape.size);
            return *this;
        }

        /**
         * 移动赋值运算符
         * 该运算符用于将一个Tensor对象的资源转移到另一个Tensor对象。
         * 它会先检查自赋值的情况，然后将源Tensor的资源转移到目标Tensor，并将源Tensor的数据指针置为nullptr。
         * 这样可以避免不必要的内存分配，提高性能。
         */
        Tensor<T> &operator=(Tensor<T> &&tensor) noexcept
        {
            if (this == &tensor)
                return *this;
            shape = tensor.shape;
            device = tensor.device;
            newer = tensor.newer;
            deleter = tensor.deleter;
            copyer = tensor.copyer;
            if (data != nullptr)
            {
                deleter(data);
            }
            data = tensor.data;
            tensor.data = nullptr;
            tensor.deleter = nullptr;
            tensor.copyer = nullptr;
            tensor.newer = nullptr;
            return *this;
        }
    }; 
}
#endif