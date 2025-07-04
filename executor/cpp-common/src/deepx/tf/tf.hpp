#ifndef DEEPX_TF_TF_HPP
#define DEEPX_TF_TF_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>

#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/dtype.hpp"

#include "stdutil/error.hpp"
#include "stdutil/num.hpp"
#include "stdutil/string.hpp"
namespace deepx::tf
{
    using mem::MemBase;
    using namespace std;
    using namespace std::chrono;

    struct Param
    {
        TypeDef dtype;
        string textvalue;

        Param(const string &textvalue = "", const DataCategory &dt = DataCategory::Unknown, const Precision &prec = Precision::Any)
            : textvalue(textvalue), dtype(make_dtype(dt, prec)) {}

        void parse(const string &param);
        string to_string() const;
    };
    // 元数据
    struct Benchmark
    {
        int repeat = 0;
    };
    struct TFMetadata
    {
        string author;
        int id;
        system_clock::time_point created_at;
        system_clock::time_point sent_at;
        system_clock::time_point recv_at;
        Benchmark benchmark;

        string to_string() const;
        void parse(const string &str);
    };
    // TF:Tensor Function的缩写
    class TF
    {
    public:
        string name;

        string tftype;
        vector<Param> args;
        vector<Param> returns;
        // metadata
        TFMetadata metadata;

    public:
        TF() = default;
        TF(const TF &) = default;
        TF(const string text);
        TF &operator=(const TF &) = default;

        string op_name();
        virtual int run(shared_ptr<MemBase> mem, string &error)
        {
            throw NotImplementError(name);
        }
        virtual string math_formula() const;

        void parse(const string &str);
        std::string to_string(bool show_extra = false, bool show_name = true) const;
        void init(const string &opname,
                  const vector<Param> &args,
                  const vector<Param> &returns);

        template <typename T>
        T getvar(int idx, shared_ptr<MemBase> mem, bool arg = true)
        {
            vector<Param> &vars = arg ? args : returns;
            if (idx < 0)
            {
                idx = vars.size() + idx;
            }
            if (idx < 0 || idx >= vars.size())
            {
                throw std::invalid_argument("Invalid argument index");
            }
            // 处理布尔类型
            if constexpr (std::is_same<T, bool>::value)
            {
                const string &value = vars[idx].textvalue;
                // 转换为小写再判断
                string lower_value = value;
                std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                               [](unsigned char c)
                               { return std::tolower(c); });

                if (lower_value == "true")
                {
                    return true;
                }
                else if (lower_value == "false")
                {
                    return false;
                }
                else
                {
                    throw std::invalid_argument("Invalid bool value:" + value);
                }
            }
            if (is_float(vars[idx].textvalue))
            {
                T value = T(std::stof(vars[idx].textvalue));
                return value;
            }
            return mem->getarg<T>(vars[idx].textvalue);
        }

        template <typename T>
        vector<T> getvector(int idx, bool arg = true)
        {
            vector<Param> &vars = arg ? args : returns;
            if (idx < 0)
            {
                idx = vars.size() + idx;
            }
            if (idx < 0 || idx >= vars.size())
            {
                throw std::invalid_argument("Invalid argument index");
            }
            if (idx < 0 || idx >= vars.size())
            {
                throw std::invalid_argument("Invalid argument index");
            }

            vector<T> result;
            string textvalue = vars[idx].textvalue;
            stdutil::trim(textvalue, "[]");
            if (textvalue.empty())
            {
                throw std::invalid_argument("Invalid argument index");
            }
            std::stringstream ss(textvalue);
            std::string item;
            while (std::getline(ss, item, ' '))
            {
                result.push_back(to<T>(item));
            }
            return result;
        }

        bool checktensors(const vector<string> &names, shared_ptr<MemBase> mem, string &error)
        {
            for (const auto &name : names)
            {
                if (!mem->existstensor(name))
                {
                    error = "tensor not found: " + name;
                    return false;
                }
            }
            return true;
        }

        std::string dtypes() const;
        bool check_dtype(const TF &other) const;

        // 添加虚拟克隆方法
        virtual shared_ptr<TF> clone() const
        {
            return make_shared<TF>(*this);
        }
    };

    class OpResp
    {
    public:
        int id;
        string result;
        system_clock::time_point recv_at;
        system_clock::time_point start_at;
        system_clock::time_point finish_at;
        string message;

    public:
        OpResp() = default;
        OpResp(const OpResp &) = default;
        OpResp &operator=(const OpResp &) = default;

        std::string to_string() const
        {
            std::stringstream stream;
            stream << id << " " << result;
            stream << "// recv_at=";
            stream << duration_cast<milliseconds>(recv_at.time_since_epoch()).count();
            stream << " start_at=";
            stream << duration_cast<milliseconds>(start_at.time_since_epoch()).count();
            stream << " finish_at=";
            stream << duration_cast<milliseconds>(finish_at.time_since_epoch()).count();
            if (message.size() > 0)
            {
                stream << " " << message;
            }
            return stream.str();
        }
        void init(int id, system_clock::time_point recv_at)
        {
            this->id = id;
            this->recv_at = recv_at;
        }
        void finish(const string &message)
        {
            this->result = "ok";
            this->finish_at = system_clock::now();
            this->message = message;
        }
        void error(const string &message)
        {
            this->result = "error";
            this->finish_at = system_clock::now();
            this->message = message;
        }
    };

}
#endif