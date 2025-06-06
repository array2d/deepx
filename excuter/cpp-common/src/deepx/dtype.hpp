#ifndef DEEPX_DTYPE_HPP
#define DEEPX_DTYPE_HPP

#include <string>
#include <cstdint>
#include <sstream>
#include <vector>
#include "stdutil/string.hpp"
#include "stdutil/num.hpp"

namespace deepx
{
    using namespace std;

    template <typename T>
    T to(const std::string &textvalue)
    {
        if constexpr (std::is_same_v<T, std::string>)
        {
            return textvalue;
        }
        else if constexpr (std::is_arithmetic_v<T>)
        {
            return static_cast<T>(std::stof(textvalue));
        }
        else
        {
            // 对于其他类型，尝试从字符串转换
            T value;
            std::istringstream iss(textvalue);
            iss >> value;
            return value;
        }
    }

    enum class DataCategory : uint8_t
    {
        Unknown = 0,
        Var = 1 << 0,        // 变量类型
        Vector = 1 << 1,     // 向量类型
        Tensor = 1 << 2,     // 张量类型
        ListTensor = 1 << 3, // 张量列表类型
        // 4-15预留
    };

    // 在DataCategory枚举定义后添加位运算操作符
    inline DataCategory operator|(DataCategory a, DataCategory b)
    {
        return static_cast<DataCategory>(
            static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    inline DataCategory operator&(DataCategory a, DataCategory b)
    {
        return static_cast<DataCategory>(
            static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

    // 修改base_category_str函数以支持组合类型
    inline std::string base_category_str(DataCategory category)
    {
        std::vector<std::string> types;
        uint8_t value = static_cast<uint8_t>(category);

        if (value & static_cast<uint8_t>(DataCategory::Tensor))
            types.push_back("tensor");
        if (value & static_cast<uint8_t>(DataCategory::Vector))
            types.push_back("vector");
        if (value & static_cast<uint8_t>(DataCategory::Var))
            types.push_back("var");
        if (value & static_cast<uint8_t>(DataCategory::ListTensor))
            types.push_back("listtensor");

        if (types.empty())
            return "unknown";

        std::string result = types[0];
        for (size_t i = 1; i < types.size(); i++)
        {
            result += "|" + types[i];
        }
        return result;
    }

    // 修改base_category函数以支持组合类型
    inline DataCategory base_category(const std::string &str)
    {
        if (str.find('|') == std::string::npos)
        {
            // 处理单一类型
            if (str == "tensor")
                return DataCategory::Tensor;
            else if (str == "vector")
                return DataCategory::Vector;
            else if (str == "var")
                return DataCategory::Var;
            else if (str == "listtensor")
                return DataCategory::ListTensor;
            return DataCategory::Unknown;
        }

        // 处理组合类型
        DataCategory result = DataCategory::Unknown;
        size_t start = 0;
        size_t pos;

        while ((pos = str.find('|', start)) != std::string::npos)
        {
            std::string type = str.substr(start, pos - start);
            result = result | base_category(type);
            start = pos + 1;
        }

        // 处理最后一个类型
        result = result | base_category(str.substr(start));
        return result;
    }

    // 将Precision改为位图形式
    enum class Precision : uint16_t
    {
        // 浮点类型 (0-7位)
        Float64 = 1 << 0,    // 0000 0000 0000 0001
        Float32 = 1 << 1,    // 0000 0000 0000 0010
        Float16 = 1 << 2,    // 0000 0000 0000 0100  // E5M10B15
        BFloat16 = 1 << 3,   // 0000 0000 0000 1000  // E8M7B127
        Float8E5M2 = 1 << 4, // 0000 0000 0001 0000  // E5M2B15
        Float8E4M3 = 1 << 5, // 0000 0000 0010 0000  // E4M3B7
        Float4E2M1 = 1 << 6, // 0000 0000 0100 0000  // E2M1B3

        // 整型 (8-12位)
        Int64 = 1 << 8,  // 0000 0001 0000 0000
        Int32 = 1 << 9,  // 0000 0010 0000 0000
        Int16 = 1 << 10, // 0000 0100 0000 0000
        Int8 = 1 << 11,  // 0000 1000 0000 0000
        Int4 = 1 << 12,  // 0001 0000 0000 0000

        // 布尔类型 (13位)
        Bool = 1 << 13,   // 0010 0000 0000 0000
        String = 1 << 15, // 0100 0000 0000 0000
                          // 常用组合
        Any = 0xFFFF,     // 1111 1111 1111 1111
        Float = Float64 | Float32 | Float16 | BFloat16 | Float8E5M2 | Float8E4M3 | Float4E2M1,
        Float8 = Float8E5M2 | Float8E4M3, // 所有FP8格式
        Int = Int64 | Int32 | Int16 | Int8 | Int4
    };

    // 添加位运算操作符
    inline Precision operator|(Precision a, Precision b)
    {
        return static_cast<Precision>(
            static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
    }

    inline Precision operator&(Precision a, Precision b)
    {
        return static_cast<Precision>(
            static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
    }
    // 在Precision枚举定义后添加位数获取函数
    inline constexpr int precision_bits(Precision p)
    {
        switch (p)
        {
        case Precision::Float64:
            return 64;
        case Precision::Float32:
            return 32;
        case Precision::Float16:
            return 16;
        case Precision::BFloat16:
            return 16;
        case Precision::Float8E5M2:
            return 8;
        case Precision::Float8E4M3:
            return 8;
        //TODO 需要根据平台支持
        // case Precision::Float4E2M1:
        //     return 4;
        case Precision::Int64:
            return 64;
        case Precision::Int32:
            return 32;
        case Precision::Int16:
            return 16;
        case Precision::Int8:
            return 8;
        //TODO，int4 需要根据平台支持
        // case Precision::Int4:
        //     return 4;
        case Precision::Bool:
            return 8;
        case Precision::String:
        case Precision::Any:
        default:
            return 0;
        }
    }

    // 删除DataCategory，直接在DataType中使用BaseCategory
    union TypeDef
    {
        struct
        {
            DataCategory category : 8; // 基础类型
            Precision precision : 16;  // 精度类型
            uint8_t reserved : 8;      // 保留位
        } parts;
        uint32_t value; // 整体访问

        // 构造函数
        constexpr TypeDef() : value(0) {}

        // 修改构造函数，使用初始化列表
        constexpr TypeDef(DataCategory c, Precision p) : value(0)
        {
            parts.category = c;
            parts.precision = p;
        }

        bool operator==(const TypeDef &other) const
        {
            return value == other.value;
        }

        bool operator!=(const TypeDef &other) const
        {
            return value != other.value;
        }

        // 判断other是否在当前类型的精度范围内
        bool match(const TypeDef &other) const
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
    };

    // 辅助函数用于创建DataType
    constexpr TypeDef make_dtype(DataCategory category, Precision precision)
    {
        return TypeDef(category, precision);
    }

    // 修改precision_str函数以使用标准命名格式
    inline std::string precision_str(Precision p)
    {
        if (p == Precision::Any)
            return "any";

        std::vector<std::string> types;
        uint16_t value = static_cast<uint16_t>(p);

        if (value & static_cast<uint16_t>(Precision::Float64))
            types.push_back("float64");
        if (value & static_cast<uint16_t>(Precision::Float32))
            types.push_back("float32");
        if (value & static_cast<uint16_t>(Precision::Float16))
            types.push_back("float16"); // 改回float16
        if (value & static_cast<uint16_t>(Precision::BFloat16))
            types.push_back("bfloat16"); // 改回bfloat16
        if (value & static_cast<uint16_t>(Precision::Float8E5M2))
            types.push_back("float8e5m2");
        if (value & static_cast<uint16_t>(Precision::Float8E4M3))
            types.push_back("float8e4m3");
        if (value & static_cast<uint16_t>(Precision::Float4E2M1))
            types.push_back("float4e2m1");
        if (value & static_cast<uint16_t>(Precision::Int64))
            types.push_back("int64");
        if (value & static_cast<uint16_t>(Precision::Int32))
            types.push_back("int32");
        if (value & static_cast<uint16_t>(Precision::Int16))
            types.push_back("int16");
        if (value & static_cast<uint16_t>(Precision::Int8))
            types.push_back("int8");
        if (value & static_cast<uint16_t>(Precision::Int4))
            types.push_back("int4");
        if (value & static_cast<uint16_t>(Precision::Bool))
            types.push_back("bool");
        if (value & static_cast<uint16_t>(Precision::String))
            types.push_back("string");
        if (types.empty())
            return "any";

        std::string result = types[0];
        for (size_t i = 1; i < types.size(); i++)
        {
            result += "|" + types[i];
        }
        return result;
    }

    // 修改dtype_str函数
    inline std::string dtype_str(const TypeDef &dtype)
    {
        return base_category_str(dtype.parts.category) +
               "<" + precision_str(dtype.parts.precision) + ">";
    }

    // 修改precision函数以匹配新的命名格式
    inline Precision precision(const std::string &str)
    {
        if (str == "any")
            return Precision::Any;
        else if (str == "float64")
            return Precision::Float64;
        else if (str == "float32")
            return Precision::Float32;
        else if (str == "float16")
            return Precision::Float16;
        else if (str == "bfloat16")
            return Precision::BFloat16;
        else if (str == "float8e5m2")
            return Precision::Float8E5M2;
        else if (str == "float8e4m3")
            return Precision::Float8E4M3;
        else if (str == "float4e2m1")
            return Precision::Float4E2M1;

        // 添加组合类型支持
        else if (str == "int")
            return Precision::Int;
        else if (str == "float")
            return Precision::Float;
        else if (str == "float8")
            return Precision::Float8;

        else if (str == "int64")
            return Precision::Int64;
        else if (str == "int32")
            return Precision::Int32;
        else if (str == "int16")
            return Precision::Int16;
        else if (str == "int8")
            return Precision::Int8;
        else if (str == "int4")
            return Precision::Int4;

        else if (str == "bool")
            return Precision::Bool;
        else if (str == "string")
            return Precision::String;
        return Precision::Any;
    }

    // 修改dtype函数，处理无精度标记的情况
    inline TypeDef dtype(const std::string &str)
    {
        size_t pos_start = str.find('<');
        size_t pos_end = str.find('>');

        if (pos_start == std::string::npos || pos_end == std::string::npos)
        {
            // 无精度标记时，使用Any作为默认精度
            return make_dtype(base_category(str), Precision::Any);
        }

        std::string category_str = str.substr(0, pos_start);
        std::string precision_str = str.substr(pos_start + 1, pos_end - pos_start - 1);

        return make_dtype(
            base_category(category_str),
            precision(precision_str));
    }

    inline TypeDef autodtype(const std::string &param)
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
            return dtype(type);
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
                        return make_dtype(DataCategory::Vector, Precision::Int32);
                    }
                    else if (is_float(vectorvalues[0]))
                    {
                        return make_dtype(DataCategory::Vector, Precision::Float64);
                    }
                    else
                    {
                        return make_dtype(DataCategory::ListTensor, Precision::Any);
                    }
                }
                else
                {
                    return make_dtype(DataCategory::Vector, Precision::Any);
                }
            }
            else
            {
                return make_dtype(DataCategory::Var | DataCategory::Tensor, Precision::Any);
            }
        }
    }

    template <Precision P>
    struct PrecisionWrapper {};

    template <typename PrecisionWrapper>
    struct to_tensor_type;

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Float64>> {
        using type = double;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Float32>> {
        using type = float;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Int64>> {
        using type = int64_t;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Int32>> {
        using type = int32_t;
    };
    
    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Int16>> {
        using type = int16_t;
    };

    template <>
    struct to_tensor_type<PrecisionWrapper<Precision::Int8>> {
        using type = int8_t;
    };

    template <Precision p>
    using tensor_t = typename to_tensor_type<PrecisionWrapper<p>>::type; 
} // namespace deepx
#endif
