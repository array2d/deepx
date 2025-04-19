#ifndef DEEPX_TF_ARG_HPP
#define DEEPX_TF_ARG_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include <any>
#include <sstream>

namespace deepx::tf
{
    class ArgSet : public TF
    {
    public:
        ArgSet(vector<Param> args, vector<Param> returns)
        {
            this->name = "argset";
            this->author = "";
            this->tftype = "arg";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "var argname = argvalue";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            if (this->args.size() != 1)
            {
                error = "argset(int32) must have 1 argument";
                return 1;
            }
            TypeDef datatype = this->returns[0].dtype;
            if (uint8_t(datatype.category() & DataCategory::Var) == 0)
            {
                error = "datatype must be var";
                return 1;
            }
            switch (datatype.precision())
            {
            case Precision::Int32:
            {
                int value = atoi(this->args[0].textvalue.c_str());
                mem->addarg(name, value);
                break;
            }
            case Precision::Float32:
            {
                float value = stof(this->args[0].textvalue.c_str());
                mem->addarg(name, value);
                break;
            }
            case Precision::Float64:
            {
                double value = stod(this->args[0].textvalue.c_str());
                mem->addarg(name, value);
                break;
            }
            default:
                error = "Unsupportednone dtype: " + dtype_str(this->args[0].dtype);
                return 1;
            }
            return 0;
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<ArgSet>(*this);
        }
    };

    class VecSet : public TF
    {
    public:
        VecSet(vector<Param> args, vector<Param> returns)
        {
            this->name = "vecset";
            this->author = "";
            this->tftype = "arg";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "shape = [3  4  5]";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->returns[0].textvalue;
            TypeDef datatype = this->returns[0].dtype;
            if (uint8_t(datatype.category() & DataCategory::Vector) == 0)
            {
                error = "datatype must be vector";
                return 1;
            }

            // 分割文本值为字符串向量
            vector<string> value_strs;
            stringstream ss(this->args[0].textvalue);
            string item;
            while (ss >> item) {
                value_strs.push_back(item);
            }

            // 根据精度类型转换并存储
            switch (datatype.precision())
            {
            case Precision::Int64:
            {
                vector<int64_t> values;
                for (const auto &str : value_strs) {
                    values.push_back(stoll(str));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Int32:
            {
                vector<int32_t> values;
                for (const auto &str : value_strs) {
                    values.push_back(stoi(str));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Int16:
            {
                vector<int16_t> values;
                for (const auto &str : value_strs) {
                    values.push_back(static_cast<int16_t>(stoi(str)));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Int8:
            {
                vector<int8_t> values;
                for (const auto &str : value_strs) {
                    values.push_back(static_cast<int8_t>(stoi(str)));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Float64:
            {
                vector<double> values;
                for (const auto &str : value_strs) {
                    values.push_back(stod(str));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Float32:
            {
                vector<float> values;
                for (const auto &str : value_strs) {
                    values.push_back(stof(str));
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::Bool:
            {
                vector<bool> values;
                for (const auto &str : value_strs) {
                    values.push_back(str == "true" || str == "1");
                }
                mem->addvector(name, values);
                break;
            }
            case Precision::String:
            {
                mem->addvector(name, value_strs);
                break;
            }
            case Precision::Float16:
            case Precision::BFloat16:
            case Precision::Float8E5M2:
            case Precision::Float8E4M3:
            case Precision::Float4E2M1:
            case Precision::Int4:
            {
                error = "Unsupported precision type: " + precision_str(datatype.precision());
                return 1;
            }
            default:
                error = "Unsupported dtype: " + dtype_str(datatype);
                return 1;
            }
            return 0;
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<VecSet>(*this);
        }
    };
}
#endif
