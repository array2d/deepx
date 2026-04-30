#ifndef DEEPX_DTYPE_DATA_CATEGORY_HPP
#define DEEPX_DTYPE_DATA_CATEGORY_HPP

#include <string>
#include <cstdint>
#include <sstream>
#include <vector>
namespace deepx
{
    using namespace std;

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
    inline std::string base_category_to_string(DataCategory category)
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
    inline DataCategory base_category_from_string(const std::string &str)
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
            result = result | base_category_from_string(type);
            start = pos + 1;
        }

        // 处理最后一个类型
        result = result | base_category_from_string(str.substr(start));
        return result;
    }
}

#endif