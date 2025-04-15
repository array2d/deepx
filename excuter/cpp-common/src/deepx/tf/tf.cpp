#include <iomanip>
#include <sstream>
#include <ctime>

#include "deepx/tf/tf.hpp"
#include "stdutil/time.hpp"
namespace deepx::tf
{
    std::pair<string, string> split_body_metadata(const string &input)
    {
        size_t meta_pos = input.find("//");
        string body = input.substr(0, meta_pos);
        string meta = (meta_pos != string::npos) ? input.substr(meta_pos + 2) : "";
        return {body, meta};
    }

    std::tuple<string, string, string> split_func_input_output(const string &body)
    {
        size_t arrow_pos = body.find("->");
        if (arrow_pos == string::npos)
        {
            throw runtime_error("Invalid IR format: missing arrow");
        }

        // 获取输入和输出部分的原始字符串
        string input_part = body.substr(0, arrow_pos);
        string output_part = body.substr(arrow_pos + 2);

        // 提取函数名 - 修改这部分逻辑
        size_t space_pos = input_part.find(' ');
        size_t paren_pos = input_part.find('(');
        size_t name_end;

        if (paren_pos != string::npos && (space_pos == string::npos || paren_pos < space_pos))
        {
            // 如果有括号且括号在空格之前,使用括号位置
            name_end = paren_pos;
        }
        else
        {
            // 否则使用空格位置或字符串末尾
            name_end = space_pos != string::npos ? space_pos : input_part.length();
        }
        string func_name = input_part.substr(0, name_end);

        // 处理输入部分，去掉函数名
        input_part = input_part.substr(name_end);

        // 处理输入部分的括号
        size_t input_paren_start = input_part.find('(');
        size_t input_paren_end = input_part.rfind(')');
        if (input_paren_start != string::npos && input_paren_end != string::npos)
        {
            // 如果有括号，去掉括号
            input_part = input_part.substr(input_paren_start + 1, input_paren_end - input_paren_start - 1);
        }
        else
        {
            // 如果没有括号，去掉可能的前导空格
            if (!input_part.empty() && input_part[0] == ' ')
            {
                input_part = input_part.substr(1);
            }
        }

        // 处理输出部分的括号
        size_t output_paren_start = output_part.find('(');
        size_t output_paren_end = output_part.rfind(')');
        if (output_paren_start != string::npos && output_paren_end != string::npos)
        {
            // 如果有括号，去掉括号
            output_part = output_part.substr(output_paren_start + 1, output_paren_end - output_paren_start - 1);
        }

        return {func_name, input_part, output_part};
    }

    // 解析单个值为具体C++类型
    any parse_single_value(const string &value_str, const TypeDef &dtype)
    {
        // 如果是字符串类型，直接返回
        if (dtype.precision() == Precision::String)
        {
            return value_str;
        }

        // 处理布尔类型
        if (dtype.precision() == Precision::Bool)
        {
            if (value_str == "true" || value_str == "1")
                return true;
            if (value_str == "false" || value_str == "0")
                return false;
            throw runtime_error("Invalid boolean value: " + value_str);
        }

        try
        {
            // 处理整数类型
            if (uint16_t(dtype.precision() & Precision::Int))
            {
                if (dtype.precision() == Precision::Int64)
                    return stoll(value_str);
                if (dtype.precision() == Precision::Int32)
                    return stoi(value_str);
                if (dtype.precision() == Precision::Int16)
                    return (int16_t)stoi(value_str);
                if (dtype.precision() == Precision::Int8)
                    return (int8_t)stoi(value_str);
                if (dtype.precision() == Precision::Int4)
                    return (int8_t)(stoi(value_str) & 0x0F);
            }

            // 处理浮点类型
            if (uint16_t(dtype.precision() & Precision::Float))
            {
                if (dtype.precision() == Precision::Float64)
                    return stod(value_str);
                if (dtype.precision() == Precision::Float32)
                    return stof(value_str);
                // Float16等其他浮点类型先转为float32
                return stof(value_str);
            }

            // Any类型，尝试按照以下顺序解析：int64 -> float64 -> bool -> string
            try
            {
                return stoll(value_str);
            }
            catch (...)
            {
                try
                {
                    return stod(value_str);
                }
                catch (...)
                {
                    if (value_str == "true")
                        return true;
                    if (value_str == "false")
                        return false;
                    return value_str;
                }
            }
        }
        catch (const std::exception &e)
        {
            throw runtime_error("Failed to parse value '" + value_str + "' as " +
                                base_category_str(dtype.category()) + "<" +
                                precision_str(dtype.precision()) + ">");
        }

        return value_str; // 默认作为字符串处理
    }

    // 主参数解析函数,param已经去除两侧空格
    Param parse_param(const string &param)
    {
        Param result;
        if (param.empty())
        {
            return result;
        }

        // 初始化变量
        string type;
        string textvalue;
        vector<string> vectorvalues;
        bool vectorvalue = false;

        // 首先检查是否包含冒号分隔符
        size_t colon_pos = param.find(':');
        if (colon_pos != string::npos)
        {
            // 使用冒号分隔类型和值
            type = param.substr(0, colon_pos);
            textvalue = param.substr(colon_pos + 1);

            // 去除type两端的空格
            type.erase(0, type.find_first_not_of(" "));
            type.erase(type.find_last_not_of(" ") + 1);

            // 去除textvalue两端的空格
            textvalue.erase(0, textvalue.find_first_not_of(" "));
            textvalue.erase(textvalue.find_last_not_of(" ") + 1);

            // 检查是否是向量值（以[开头，]结尾）
            if (textvalue.size() >= 2 && textvalue.front() == '[' && textvalue.back() == ']')
            {
                vectorvalue = true;
                // 去除方括号，只保留内容
                textvalue = textvalue.substr(1, textvalue.length() - 2);
            }
        }
        else
        {
            // 无冒号，使用原有逻辑处理
            // 检查是否是向量值（以]结尾）
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
                    type.erase(0, type.find_first_not_of(" "));
                    type.erase(type.find_last_not_of(" ") + 1);
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
            result.dtype = dtype(type);
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
                        result.dtype = make_dtype(DataCategory::Vector, Precision::Int32);
                    }
                    else if (is_float(vectorvalues[0]))
                    {
                        result.dtype = make_dtype(DataCategory::Vector, Precision::Float64);
                    }
                    else
                    {
                        result.dtype = make_dtype(DataCategory::ListTensor, Precision::Any);
                    }
                }
                else
                {
                    result.dtype = make_dtype(DataCategory::Vector, Precision::Any);
                }
            }
            else
            {
                result.dtype = make_dtype(DataCategory::Var | DataCategory::Tensor, Precision::Any);
            }
        }

        result.textvalue = textvalue;
        return result;
    };

    // 解析参数列表
    vector<Param> parse_params(const string &params_str)
    {
        vector<Param> params;
        stringstream params_ss(params_str);
        string param;

        while (getline(params_ss, param, ','))
        {
            // 去除首尾空格
            param.erase(0, param.find_first_not_of(" "));
            param.erase(param.find_last_not_of(" ") + 1);

            if (param.empty())
                continue;

            // 解析单个参数
            Param parsed_param = parse_param(param);
            params.push_back(parsed_param);
        }

        return params;
    }

    // 解析元数据键值对
    void parse_metadata_pair(const string &key_value, int &id, string &author,
                             system_clock::time_point &created_at,
                             system_clock::time_point &sent_at)
    {
        size_t eq_pos = key_value.find('=');
        if (eq_pos == string::npos)
            return;

        string key = key_value.substr(0, eq_pos);
        string value = key_value.substr(eq_pos + 1);

        if (key == "id")
        {
            id = stoi(value);
        }
        else if (key == "author")
        {
            author = value;
        }
        else if (key == "created_at")
        {
            created_at = system_clock::from_time_t(stod(value));
        }
        else if (key == "sent_at")
        {
            sent_at = system_clock::from_time_t(stod(value));
        }
    }

    // 解析元数据
    void parse_metadata(const string &meta, int &id, string &author,
                        system_clock::time_point &created_at,
                        system_clock::time_point &sent_at)
    {
        if (meta.empty())
            return;

        stringstream meta_ss(meta);
        string key_value;
        while (meta_ss >> key_value)
        {
            parse_metadata_pair(key_value, id, author, created_at, sent_at);
        }
    }

    // 主解析函数
    void TF::parse(const string &input)
    {
        // 1. 按//分割主体和元数据
        auto [body, meta] = split_body_metadata(input);

        // 2. 按->分割输入和输出，同时获取函数名
        auto [func_name, input_part, output_part] = split_func_input_output(body);
        name = func_name;

        // 3. 解析输入为参数列表
        args = parse_params(input_part);

        // 4. 解析返回值
        returns = parse_params(output_part);

        // 5. 解析元数据
        parse_metadata(meta, id, author, created_at, sent_at);
    }

    void TF::init(const string &opname,
                  const vector<Param> &args,
                  const vector<Param> &returns)
    {
        this->name = opname;
        this->author = "";
        this->args = args;
        this->returns = returns;
    }

    std::string TF::to_string(bool show_extra, bool show_name) const
    {
        std::stringstream ss;
        ss << name << "(";

        // 处理输入参数
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (i > 0)
            {
                ss << ", "; // 始终使用逗号分隔参数
            }

            // 输出类型，根据show_name决定是否输出参数名
            ss << dtype_str(args[i].dtype);
            if (show_name)
            {
                ss << " " << args[i].textvalue;
            }
        }

        ss << ")->(";

        // 处理返回值
        for (size_t i = 0; i < returns.size(); ++i)
        {
            if (i > 0)
            {
                ss << ", "; // 始终使用逗号分隔返回值
            }

            // 输出类型，根据show_name决定是否输出返回值名
            ss << dtype_str(returns[i].dtype);
            if (show_name)
            {
                ss << " " << returns[i].textvalue;
            }
        }

        ss << ")";

        if (show_extra)
        {
            ss << " //id=" << id
               << " created_at=" << stdutil::format_time(created_at)
               << " sent_at=" << stdutil::format_time(sent_at);
        }

        return ss.str();
    }

    TF::TF(const string text)
    {
        parse(text);
    }
    string TF::op_name()
    {
        return name;
    }

    string TF::math_formula() const
    {
        return "";
    }

    std::string TF::dtypes() const
    {
        std::string full = to_string(false, false);
        size_t start = full.find('(');
        if (start != std::string::npos)
        {
            return full.substr(start);
        }
        return full;
    }

    bool TF::check_dtype(const TF &other) const
    {
        // 检查参数数量是否匹配
        if (args.size() != other.args.size() || returns.size() != other.returns.size())
        {
            return false;
        }

        // 检查每个参数的类型
        for (size_t i = 0; i < args.size(); ++i)
        {
            // 当前TF的类型可能包含多个选项
            TypeDef dtype = args[i].dtype;
            TypeDef other_dtype = other.args[i].dtype;
            // TODO
        }

        return true;
    }

}