
#ifndef DEEPX_DTYPE_PRECISION_HPP
#define DEEPX_DTYPE_PRECISION_HPP

#include <string>
#include <cstdint>
#include <sstream>
#include <vector>
namespace deepx
{
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
        // TODO 需要根据平台支持
        //  case Precision::Float4E2M1:
        //      return 4;
        case Precision::Int64:
            return 64;
        case Precision::Int32:
            return 32;
        case Precision::Int16:
            return 16;
        case Precision::Int8:
            return 8;
        // TODO，int4 需要根据平台支持
        //  case Precision::Int4:
        //      return 4;
        case Precision::Bool:
            return 8;
        case Precision::String:
        case Precision::Any:
        default:
            return 0;
        }
    }

    // 修改precision函数以匹配新的命名格式
    inline Precision precision_from_string(const std::string &str)
    {
        if (str == "any")
            return Precision::Any;
        else if (str == "float64" || str == "f64")
            return Precision::Float64;
        else if (str == "float32" || str == "f32")
            return Precision::Float32;
        else if (str == "float16" || str == "f16")
            return Precision::Float16;
        else if (str == "bfloat16" || str == "bf16")
            return Precision::BFloat16;
        else if (str == "float8e5m2" || str == "f8e5m2")
            return Precision::Float8E5M2;
        else if (str == "float8e4m3" || str == "f8e4m3")
            return Precision::Float8E4M3;
        else if (str == "float4e2m1" || str == "f4e2m1")
            return Precision::Float4E2M1;

        // 添加组合类型支持
        else if (str == "int")
            return Precision::Int;
        else if (str == "float")
            return Precision::Float;
        else if (str == "float8")
            return Precision::Float8;

        else if (str == "int64" || str == "i64")
            return Precision::Int64;
        else if (str == "int32" || str == "i32")
            return Precision::Int32;
        else if (str == "int16" || str == "i16")
            return Precision::Int16;
        else if (str == "int8" || str == "i8")
            return Precision::Int8;
        else if (str == "int4" || str == "i4")
            return Precision::Int4;

        else if (str == "bool")
            return Precision::Bool;
        else if (str == "string")
            return Precision::String;
        return Precision::Any;
    }



    // 修改precision_str函数以使用标准命名格式
    inline std::string precision_to_string(Precision p)
    {
        if (p == Precision::Any)
            return "any";

        std::vector<std::string> types;
        uint16_t value = static_cast<uint16_t>(p);

        if (value & static_cast<uint16_t>(Precision::Float64))
            types.push_back("f64");
        if (value & static_cast<uint16_t>(Precision::Float32))
            types.push_back("f32");
        if (value & static_cast<uint16_t>(Precision::Float16))
            types.push_back("f16");
        if (value & static_cast<uint16_t>(Precision::BFloat16))
            types.push_back("bf16");
        if (value & static_cast<uint16_t>(Precision::Float8E5M2))
            types.push_back("f8e5m2");
        if (value & static_cast<uint16_t>(Precision::Float8E4M3))
            types.push_back("f8e4m3");
        if (value & static_cast<uint16_t>(Precision::Float4E2M1))
            types.push_back("f4e2m1");
        if (value & static_cast<uint16_t>(Precision::Int64))
            types.push_back("i64");
        if (value & static_cast<uint16_t>(Precision::Int32))
            types.push_back("i32");
        if (value & static_cast<uint16_t>(Precision::Int16))
            types.push_back("i16");
        if (value & static_cast<uint16_t>(Precision::Int8))
            types.push_back("i8");
        if (value & static_cast<uint16_t>(Precision::Int4))
            types.push_back("i4");
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

}
#endif