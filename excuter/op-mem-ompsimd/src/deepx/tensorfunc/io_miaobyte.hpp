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

    template <typename T>
    void save(Tensor<T> &tensor, const std::string &path)
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
        int data_size = tensor.shape.size * precision_bits(tensor.shape.dtype) / 8;
        data_fs.write(reinterpret_cast<const char *>(tensor.data), data_size);
        data_fs.close();
    }

    //load


    template <typename T>
    pair<std::string,shared_ptr<Tensor<T>>> load(const std::string &path)
    {
        // 加载shape
        pair<std::string,Shape> shape_name=loadShape(path);
        Shape shape=shape_name.second;
        std::string tensor_name=shape_name.first;
 

        // 检查T 和 shape.dtype 是否匹配
        if (shape.dtype != precision<T>())
        {
            throw std::runtime_error("调用load<" + precision_str(shape.dtype) + "> 不匹配: 需要 " + precision_str(shape.dtype) +
                                     " 类型，但文件为" + precision_str(precision<T>()) + " 类型");
        }

        // 检查file.size，是否是tensor.size*sizeof(T)
        std::string datapath = path + ".data";
        std::ifstream data_fs(datapath, std::ios::binary);
        data_fs.seekg(0, std::ios::end);
        std::streamsize fileSize = data_fs.tellg();
        std::streamsize expectedSize = shape.size * precision_bits(shape.dtype) / 8;

        if (fileSize != expectedSize)
        {
            throw std::runtime_error("数据文件大小不足: 需要 " + std::to_string(expectedSize) +
                                     " 字节，但文件只有 " + std::to_string(fileSize) + " 字节");
        }
        data_fs.seekg(0);

        // 创建tensor
        shared_ptr<Tensor<T>> tensor = make_shared<Tensor<T>>(New<T>(shape.shape));
        data_fs.read(reinterpret_cast<char *>(tensor->data), fileSize);
        data_fs.close();
        return std::make_pair(tensor_name, tensor);
    };

}
#endif // DEEPX_TENSORFUNC_IO_MIAOBYTE_HPP