#include <iomanip>
#include <sstream>
#include <ctime>
#include <map>

#include "deepx/tf/tf.hpp"
#include "stdutil/time.hpp"
#include "stdutil/string.hpp"
namespace deepx::tf
{

    void Param::parse(const string &param)
    {
        // 1. 按:分割类型和值
        size_t colon_pos = param.find(':');
        string type, textvalue;
        if (colon_pos != string::npos)
        {
            type = param.substr(0, colon_pos);
            stdutil::trimspace(type);
            textvalue = param.substr(colon_pos + 1);
            stdutil::trimspace(textvalue);
        }
        else
        {
            textvalue = param;
            stdutil::trimspace(textvalue);
        }
        if (!type.empty())
        {
            this->dtype = deepx::dtype(type);
            this->textvalue = textvalue;
        }
        else
        {
            this->dtype = deepx::dtype(textvalue);
            this->textvalue = textvalue;
        }
    }
    string Param::to_string() const
    {
        return dtype_str(dtype) + ":" + textvalue;
    }
    string TFMetadata::to_string() const
    {
        stringstream ss;
        if (!author.empty())
        {
            ss << "author=" << author << " ";
        }
        if (id > 0)
        {
            ss << "id=" << id << " ";
        }
        if (created_at != system_clock::time_point::min())
        {
            ss << "created_at=" << duration_cast<milliseconds>(created_at.time_since_epoch()).count() << " ";
        }
        if (sent_at != system_clock::time_point::min())
        {
            ss << "sent_at=" << duration_cast<milliseconds>(sent_at.time_since_epoch()).count() << " ";
        }
        if (recv_at != system_clock::time_point::min())
        {
            ss << "recv_at=" << duration_cast<milliseconds>(recv_at.time_since_epoch()).count() << " ";
        }
        if (benchmark.repeat > 0)
        {
            ss << "benchmark.repeat=" << benchmark.repeat << " ";
        }
        return ss.str();
    }

     std::unordered_map<string, string> parse_metadata_map(const string &meta)
    {
        std::unordered_map<string, string> metadata;
        stringstream meta_ss(meta);
        string key_value;
        while (meta_ss >> key_value)
        {
            size_t eq_pos = key_value.find('=');
            if (eq_pos == string::npos)
                continue;
            string key = key_value.substr(0, eq_pos);
            string value = key_value.substr(eq_pos + 1);
            metadata[key] = value;
        }
        return metadata;
    }

    // 解析元数据
    void  TFMetadata::parse(const string &meta)
    {
        if (meta.empty())
            return;

        auto metadata_map = parse_metadata_map(meta);
        if (metadata_map.find("id") != metadata_map.end())
        {
            id = stoi(metadata_map["id"]);
        }
        if (metadata_map.find("author") != metadata_map.end())
        {
            author = metadata_map["author"];
        }   
        if (metadata_map.find("created_at") != metadata_map.end())
        {
            created_at = system_clock::from_time_t(stod(metadata_map["created_at"]));
        }
        if (metadata_map.find("sent_at") != metadata_map.end())
        {
            sent_at = system_clock::from_time_t(stod(metadata_map["sent_at"]));
        }
        if (metadata_map.find("recv_at") != metadata_map.end())
        {
            recv_at = system_clock::from_time_t(stod(metadata_map["recv_at"]));
        }
        if (metadata_map.find("benchmark.repeat") != metadata_map.end())
        {   
            benchmark.repeat = stoi(metadata_map["benchmark.repeat"]);
        }
    }

    // 分割主体和元数据
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

    // 解析参数列表
    vector<Param> parse_params(const string &params_str)
    {
        vector<Param> params;
        stringstream params_ss(params_str);
        string param;

        while (getline(params_ss, param, ','))
        {
            // 去除首尾空格
            stdutil::trimspace(param);

            if (param.empty())
                continue;

            // 解析单个参数
            Param parsed_param;
            parsed_param.parse(param);
            params.push_back(parsed_param);
        }

        return params;
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
        metadata.parse(meta);
    }

    void TF::init(const string &opname,
                  const vector<Param> &args,
                  const vector<Param> &returns)
    {
        this->name = opname;
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
            ss << args[i].to_string();
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
            ss << returns[i].to_string();
        }

        ss << ")";

        if (show_extra)
        {
            ss << " //" << metadata.to_string();
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