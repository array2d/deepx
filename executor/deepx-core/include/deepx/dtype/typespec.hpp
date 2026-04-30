#ifndef DEEPX_DTYPE_TYPEDEF_HPP
#define DEEPX_DTYPE_TYPEDEF_HPP

#include <cstdint>

#include "stdutil/string.hpp"
#include "stdutil/num.hpp"

#include "deepx/dtype/data_category.hpp"
#include "deepx/dtype/precision.hpp"


namespace deepx
{
    union TypeSpec
    {
        struct
        {
            DataCategory category : 8; // 基础类型
            Precision precision : 16;  // 精度类型
            uint8_t reserved : 8;      // 保留位
        } parts;
        uint32_t value; // 整体访问

        // 构造函数
        constexpr TypeSpec() : value(0) {}

        // 修改构造函数，使用初始化列表
        constexpr TypeSpec(DataCategory c, Precision p) : value(0)
        {
            parts.category = c;
            parts.precision = p;
        }

        bool operator==(const TypeSpec &other) const
        {
            return value == other.value;
        }

        bool operator!=(const TypeSpec &other) const
        {
            return value != other.value;
        }

        // 判断other是否在当前类型的精度范围内
        bool match(const TypeSpec &other) const
        {
            // 类型必须相同
            uint8_t this_cat = static_cast<uint8_t>(parts.category);
            uint8_t other_cat = static_cast<uint8_t>(other.parts.category);
            if ((this_cat & other_cat) != this_cat)
            {
                return false;
            }

            // 使用位操作检查precision
            // 检查this的precision位是否都在other的precision中
            uint16_t this_prec = static_cast<uint16_t>(parts.precision);
            uint16_t other_prec = static_cast<uint16_t>(other.parts.precision);
            return (this_prec & other_prec) == this_prec;
        }
        constexpr DataCategory category() const
        {
            return parts.category;
        }

        constexpr Precision precision() const
        {
            return parts.precision;
        }
        string to_string() const
        {
            return  base_category_to_string(parts.category) +
               "<" + precision_to_string(parts.precision) + ">";
        }
        void from_string(const string &str)
        {
            size_t pos_start = str.find('<');
            size_t pos_end = str.find('>');

            if (pos_start == std::string::npos || pos_end == std::string::npos)
            {
                parts.category = base_category_from_string(str);
                parts.precision = Precision::Any;
                return;
            }

            std::string category_str = str.substr(0, pos_start);
            std::string precision_str = str.substr(pos_start + 1, pos_end - pos_start - 1);

            parts.category = base_category_from_string(category_str);
            parts.precision = precision_from_string(precision_str);
        }
    };
    
    inline TypeSpec typespec_from_string(const std::string &str){
        TypeSpec ts;
        ts.from_string(str);
        return ts;
    }
    
    inline TypeSpec autodtype(const std::string &param)
    {
        std::string type;
        std::string textvalue;
        std::vector<std::string> vectorvalues;
        bool vectorvalue = false;
        if (param.back() == ']')
        {
            size_t bracket_start = param.find('[');
            if (bracket_start != string::npos)
            {
                vectorvalue = true;
                // 提取方括号内的内容作为textvalue
                textvalue = param.substr(bracket_start + 1, param.length() - bracket_start - 2);
                // 提取方括号前的内容作为type
                type = param.substr(0, bracket_start);
                // 去除type两端的空格
                stdutil::trim(type);
            }
        }

        if (!vectorvalue)
        {
            // 没有方括号，按空格分割
            stringstream ss(param);
            string first, second;
            ss >> first;
            if (ss >> second)
            {
                // 如果能读取到两个部分
                type = first;
                textvalue = second;
            }
            else
            {
                textvalue = first;
            }
        }
        // 处理向量值
        if (vectorvalue)
        {
            // 分割字符串为向量
            stringstream ss(textvalue);
            string item;
            while (getline(ss, item, ' '))
            {
                item.erase(0, item.find_first_not_of(" "));
                item.erase(item.find_last_not_of(" ") + 1);
                if (!item.empty())
                {
                    vectorvalues.push_back(item);
                }
            }
        }

        // 设置结果
        if (!type.empty())
        {
            return typespec_from_string(type);
        }
        else
        {
            // 没有显式类型声明,根据值推断
            if (vectorvalue)
            {
                if (!vectorvalues.empty())
                {
                    if (is_integer(vectorvalues[0]))
                    {
                        return TypeSpec(DataCategory::Vector, Precision::Int32);
                    }
                    else if (is_float(vectorvalues[0]))
                    {
                        return TypeSpec(DataCategory::Vector, Precision::Float64);
                    }
                    else
                    {
                        return TypeSpec(DataCategory::ListTensor, Precision::Any);
                    }
                }
                else
                {
                    return TypeSpec(DataCategory::Vector, Precision::Any);
                }
            }
            else
            {
                return TypeSpec(DataCategory::Var | DataCategory::Tensor, Precision::Any);
            }
        }
    }

}
#endif