#ifndef DEEPX_TF_INIT_HPP
#define DEEPX_TF_INIT_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "stdutil/num.hpp"
namespace deepx::tf
{

    template <typename Author>
    class Constant : public TF
    {
    public:
        Constant(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "constant";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        Constant(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "constant")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            auto tensor = mem->gettensor(name).get();
            if (tensor==nullptr) {
                error = "tensor not found: " + name;
                return 1;
            }
            auto type = tensor->shape.dtype;
            switch (type)
            {
            case Precision::Float64:
            {
                auto output = mem->gettensor<double>(name).get();
                tensorfunc::constant<Author, double>(*output, this->getvar<double>(1, mem));
                break;
            }
            case Precision::Float32:
            {
                auto output = mem->gettensor<float>(name).get();
                tensorfunc::constant<Author, float>(*output, this->getvar<float>(1, mem));
                break;
            }

            case Precision::Int64:
            {
                auto output = mem->gettensor<int64_t>(name).get();
                tensorfunc::constant<Author, int64_t>(*output, this->getvar<int64_t>(1, mem));
                break;
            }
            case Precision::Int32:
            {
                auto output = mem->gettensor<int32_t>(name).get();
                tensorfunc::constant<Author, int32_t>(*output, this->getvar<int32_t>(1, mem));
                break;
            }
            case Precision::Int16:
            {
                auto output = mem->gettensor<int16_t>(name).get();
                tensorfunc::constant<Author, int16_t>(*output, this->getvar<int16_t>(1, mem));
                break;
            }
            case Precision::Int8:
            {
                auto output = mem->gettensor<int8_t>(name).get();
                tensorfunc::constant<Author, int8_t>(*output, this->getvar<int8_t>(1, mem));
                break;
            }
            default:
            {
                error = "unsupported dtype: " + precision_str(type);
                return 1;
            }
            }
            return 0;
        };
        string math_formula() const override
        {
            return "constant(T1,value)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Constant<Author>>(*this);
        }
    };

    template <typename Author>
    class Arange : public TF
    {
    public:
        Arange(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "arange";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            auto tensor = mem->gettensor(name).get();
            auto type = tensor->shape.dtype;
            switch (type)
            {
            case Precision::Float64:
            {
                auto output = mem->gettensor<double>(name).get();
                tensorfunc::arange<Author, double>(*output, this->getvar<double>(1, mem), this->getvar<double>(2, mem));
                break;
            }
            case Precision::Float32:
            {
                auto output = mem->gettensor<float>(name).get();
                tensorfunc::arange<Author, float>(*output, this->getvar<float>(1, mem), this->getvar<float>(2, mem));
                break;
            }

            case Precision::Int64:
            {
                auto output = mem->gettensor<int64_t>(name).get();
                tensorfunc::arange<Author, int64_t>(*output, this->getvar<int64_t>(1, mem), this->getvar<int64_t>(2, mem));
                break;
            }
            case Precision::Int32:
            {
                auto output = mem->gettensor<int32_t>(name).get();
                tensorfunc::arange<Author, int32_t>(*output, this->getvar<int32_t>(1, mem), this->getvar<int32_t>(2, mem));
                break;
            }
            case Precision::Int16:
            {
                auto output = mem->gettensor<int16_t>(name).get();
                tensorfunc::arange<Author, int16_t>(*output, this->getvar<int16_t>(1, mem), this->getvar<int16_t>(2, mem));
                break;
            }
            case Precision::Int8:
            {
                auto output = mem->gettensor<int8_t>(name).get();
                tensorfunc::arange<Author, int8_t>(*output, this->getvar<int8_t>(1, mem), this->getvar<int8_t>(2, mem));
                break;
            }
            default:
            {
                error = "unsupported dtype: " + precision_str(type);
                return 1;
            }
            }
            return 0;
        }
        string math_formula() const override
        {
            return "arange(T1,start,step)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Arange<Author>>(*this);
        }
    };

    template <typename Author>
    class Uniform : public TF
    {
    public:
        Uniform(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "uniform";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            auto tensor = mem->gettensor(name).get();
            auto type = tensor->shape.dtype;
            switch (type)
            {
            case Precision::Float64:
            {
                auto output = mem->gettensor<double>(name).get();
                tensorfunc::uniform<Author, double>(*output, this->getvar<double>(1, mem), this->getvar<double>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }
            case Precision::Float32:
            {
                auto output = mem->gettensor<float>(name).get();
                tensorfunc::uniform<Author, float>(*output, this->getvar<float>(1, mem), this->getvar<float>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }

            case Precision::Int64:
            {
                auto output = mem->gettensor<int64_t>(name).get();
                tensorfunc::uniform<Author, int64_t>(*output, this->getvar<int64_t>(1, mem), this->getvar<int64_t>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }
            case Precision::Int32:
            {
                auto output = mem->gettensor<int32_t>(name).get();
                tensorfunc::uniform<Author, int32_t>(*output, this->getvar<int32_t>(1, mem), this->getvar<int32_t>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }
            case Precision::Int16:
            {
                auto output = mem->gettensor<int16_t>(name).get();
                tensorfunc::uniform<Author, int16_t>(*output, this->getvar<int16_t>(1, mem), this->getvar<int16_t>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }
            case Precision::Int8:
            {
                auto output = mem->gettensor<int8_t>(name).get();
                tensorfunc::uniform<Author, int8_t>(*output, this->getvar<int8_t>(1, mem), this->getvar<int8_t>(2, mem), this->getvar<unsigned int>(3, mem));
                break;
            }
            default:
            {
                error = "unsupported dtype: " + precision_str(type);
                return 1;
            }
            }
            return 0;
        }
        string math_formula() const override
        {
            return "uniform(T1,low,high,seed)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Uniform<Author>>(*this);
        }
    };

}

#endif
