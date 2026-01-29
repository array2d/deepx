#ifndef STDUTIL_PRINT_HPP__
#define STDUTIL_PRINT_HPP__

#include <iostream>
#include <vector>

#include "deepx/dtype.hpp"
#include "deepx/shape.hpp"
#include "stdutil/vector.hpp"
namespace stdutil
{
    using namespace deepx;
    inline void print_element(const void *data, int offset, const deepx::Precision &dtype, const std::string &format)
    {
        switch (dtype)
        {
        case Precision::Bool:{
            bool bool_data = ((bool *)data)[offset];
            printf(format.c_str(), static_cast<int8_t>(bool_data));
            break;
        }
        case  Precision::Int8:
            printf(format.c_str(), ((int8_t *)data)[offset]);
            break;
        case Precision::Int16:
            printf(format.c_str(), ((int16_t *)data)[offset]);
            break;
        case Precision::Int32:
            printf(format.c_str(), ((int32_t *)data)[offset]);
            break;
        case Precision::Int64:
            printf(format.c_str(), ((int64_t *)data)[offset]);
            break;

        case Precision::Float64:
            printf(format.c_str(), ((double *)data)[offset]);
            break;
        case Precision::Float32:{
            float result = ((float *)data)[offset];
            printf(format.c_str(), result);
            break;
        }
        case Precision::Float16:{
            float result = ((float *)data)[offset];
            printf(format.c_str(), result);
            break;
        }   
        case Precision::BFloat16:{
            float result = ((float *)data)[offset];
            printf(format.c_str(), result);
            break;
        }
        }
    }

    inline std::string default_format(const deepx::Precision &dtype)
    {
        std::string format = "";

        if (dtype == Precision::Int4 || dtype == Precision::Int8 || dtype == Precision::Int16 || dtype == Precision::Int32 || dtype == Precision::Int64)
        {
            format = "%d";
        }
        else if (dtype == Precision::Float32 || dtype == Precision::Float64)
        {
            format = "%.4f";
        }
        else if (dtype == Precision::Bool)
        {
            format = "%d";
        }
        else if (dtype == Precision::String)
        {
            format = "%s";
        };

        return format;
    }

    void print(const std::vector<int> &shape_vec, void *data, const Precision &dtype, const std::string &f = "")
    {
        std::string format = f;
        if (f.empty())
        {
            format = stdutil::default_format(dtype);
        }

        // 创建临时Shape对象用于打印和计算
        deepx::Shape shape(shape_vec);
        shape.dtype = dtype;

        shape.print();
        if (shape.dim() == 1)
        {
            std::cout << "[";
            for (int i = 0; i < shape[0]; ++i)
            {
                if (i > 0)
                    std::cout << " ";
                stdutil::print_element(data, i, dtype, format);
            }
            std::cout << "]" << std::endl;
        }
        else
        {
            shape.range(-2, [&format, data, &shape, &dtype](const int idx_linear, const std::vector<int> &indices)
                        {
                        std::cout << indices << "=";
                        std::cout<<"["<<std::endl;
                        for (int i = 0; i < shape[-2]; ++i)
                        {
                            std::cout << " [";
                            for (int j = 0; j < shape[-1]; ++j)
                            {
                                if (j > 0)
                                    std::cout << " ";
                                int offset = idx_linear + i * shape[-1] + j;
                                print_element(data, offset, dtype, format);
                            }
                            
                            std::cout<<"]";
                            if (i<shape[-2]-1){
                                std::cout<<",";
                            }
                            std::cout<<std::endl;
                        }
                        std::cout<<"]"<<std::endl; });
        }
    }

}
#endif