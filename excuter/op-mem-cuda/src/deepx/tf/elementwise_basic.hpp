#ifndef DEEPX_TF_ELEMENTWISE_BASIC_HPP
#define DEEPX_TF_ELEMENTWISE_BASIC_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "deepx/tensorfunc/elementwise_miaobyte_basic.hpp"
#include "deepx/tensorfunc/elementwise_cublas_basic.hpp"

namespace deepx::tf
{
    template <typename Author>
    class Add : public TF
    {
    public:
        Add(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "add";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Add(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "add")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=T1+T2";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Add<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::add<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::add<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::add<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;  
            case Precision::BFloat16:
                tensorfunc::add<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::add<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::add<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::add<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::add<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class AddScalar : public TF
    {
    public:
        AddScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "addscalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        AddScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "addscalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=T1+scalar";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<AddScalar<Author>>(*this);
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
                tensorfunc::addscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue),  this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::addscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::addscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;  
            case Precision::BFloat16:
                tensorfunc::addscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::addscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::addscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::addscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::addscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Sub : public TF
    {
    public:
        Sub(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "sub";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Sub(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "sub")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=T1-T2";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Sub<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::sub<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::sub<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::sub<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;  
            case Precision::BFloat16:
                tensorfunc::sub<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::sub<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::sub<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::sub<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::sub<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class SubScalar : public TF
    {
    public: 
        SubScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "subscalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        SubScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "subscalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=T1-scalar";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<SubScalar<Author>>(*this);
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
                tensorfunc::subscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::subscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::subscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::subscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::subscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::subscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::subscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;  
            case Precision::Int8:
                tensorfunc::subscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }   
    };
 
    
};

#endif // DEEPX_TF_ELEMENTWISE_BASIC_HPP
