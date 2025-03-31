#ifndef DEEPX_TF_ELEMENTWISE_SQRT_HPP
#define DEEPX_TF_ELEMENTWISE_SQRT_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tensorfunc/elementwise_miaobyte_sqrt.hpp"

namespace deepx::tf
{

    template <typename Author>
    class Sqrt : public TF
    {
    public:
        Sqrt(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "sqrt";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Sqrt(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "sqrt")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=sqrt(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Sqrt<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::sqrt<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::sqrt<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            // case Precision::Float16:
            //     tensorfunc::sqrt<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            // case Precision::Float16:
            //     tensorfunc::sqrt<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            case Precision::Int64:
                tensorfunc::sqrt<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::sqrt<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::sqrt<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::sqrt<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Pow : public TF
    {
    public:
        Pow(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "pow";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Pow(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "pow")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=pow(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Pow<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type || b_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type) + " or " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::pow<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::pow<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            // case Precision::BFloat16:
            //     tensorfunc::pow<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            // case Precision::Float16:
            //     tensorfunc::pow<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            case Precision::Int64:
                tensorfunc::pow<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::pow<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::pow<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::pow<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class PowScalar : public TF
    {
    public:
        PowScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "powscalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        PowScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "powscalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=pow(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<PowScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type || b_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type) + " or " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::powscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue),  this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::powscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            // case Precision::BFloat16:
            //     tensorfunc::powscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            // case Precision::Float16:
            //     tensorfunc::powscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            case Precision::Int64:
                tensorfunc::powscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::powscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::powscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::powscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Log : public TF
    {
    public:
        Log(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "log";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Log(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "log")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=log(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Log<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::log<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::log<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            // case Precision::Float16:
            //     tensorfunc::log<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            // case Precision::BFloat16:
            //     tensorfunc::log<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            case Precision::Int64:
                tensorfunc::log<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::log<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::log<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::log<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Exp : public TF
    {
    public:
        Exp(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "exp";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Exp(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "exp")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=exp(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Exp<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::exp<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::exp<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            // case Precision::Float16:
            //     tensorfunc::exp<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            // case Precision::BFloat16:
            //     tensorfunc::exp<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
            //     break;
            case Precision::Int64:
                tensorfunc::exp<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::exp<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::exp<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::exp<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
};
#endif // DEEPX_TF_ELEMENTWISE_SQRT_HPP
