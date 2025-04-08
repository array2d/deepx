#ifndef DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"
#include "stdutil/print.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/io.hpp"
#include "deepx/tensorfunc/new.hpp"

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

    template <typename T>
    struct saveDispatcher<miaobyte, T>
    {
        static void save(Tensor<T> &tensor, const std::string &path, int filebegin = 0)
        {
            // 保存shape
            std::string shapepath = path + ".shape";
            std::string shapedata = tensor.shape.toYaml();
            std::ofstream shape_fs(shapepath, std::ios::binary);
            shape_fs.write(shapedata.c_str(), shapedata.size());
            shape_fs.close();

            // 保存data
            std::string datapath = path + ".data";
            std::ofstream data_fs(datapath, std::ios::binary | std::ios::in | std::ios::out);

            if (!data_fs.is_open())
            {
                // 如果文件不存在，则创建新文件
                data_fs.open(datapath, std::ios::binary | std::ios::out);
            }
            data_fs.seekp(filebegin);
            data_fs.write(reinterpret_cast<const char *>(tensor.data), tensor.shape.size * sizeof(T));
            data_fs.close();
        }
    };
    template <typename T>
    struct loadDispatcher<miaobyte, T>
    {
        static Tensor<T> load(const std::string &path, int filebegin = 0)
        {
            // 加载shape
            std::string shapepath = path + ".shape";
            std::ifstream shape_fs(shapepath, std::ios::binary);
            std::string shapedata((std::istreambuf_iterator<char>(shape_fs)), std::istreambuf_iterator<char>());

            Shape shape;
            shape.fromYaml(shapedata);
            shape_fs.close();

            // 加载data
            Tensor<T> tensor = New<T>(shape);
            std::string datapath = path + ".data";
            std::ifstream data_fs(datapath, std::ios::binary);

            if (!data_fs.is_open())
            {
                throw std::runtime_error("无法打开数据文件: " + datapath);
            }

            // 设置读取位置
            data_fs.seekg(filebegin);
            data_fs.read(reinterpret_cast<char *>(tensor.data), shape.size * sizeof(T));
            data_fs.close();

            return tensor;
        }
    };
}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP