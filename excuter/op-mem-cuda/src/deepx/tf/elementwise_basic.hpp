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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            if(!checktensors({this->args[0].textvalue,this->args[1].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            if(!checktensors({this->args[0].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
                tensorfunc::addscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            if(!checktensors({this->args[0].textvalue,this->args[1].textvalue,this->returns[0].textvalue},mem, error))  
            {
                return 1;
            }
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            if(!checktensors({this->args[0].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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

    template <typename Author>
    class Mul : public TF
    {
    public:
        Mul(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "mul";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=T1*T2";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Mul<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue,this->args[1].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
                tensorfunc::mul<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::mul<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::mul<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::mul<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::mul<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::mul<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::mul<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::mul<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class MulScalar : public TF
    {
    public:
        MulScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "mulscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=T1*scalar";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<MulScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
                tensorfunc::mulscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::mulscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::mulscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::mulscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::mulscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::mulscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::mulscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::mulscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Div : public TF
    {
    public:
        Div(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "div";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=T1/T2";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Div<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue,this->args[1].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
                tensorfunc::div<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::div<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::div<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::div<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::div<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::div<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::div<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::div<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class DivScalar : public TF
    {
    public:
        DivScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "divscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=scalar/T1";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<DivScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
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
                tensorfunc::divscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::divscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::divscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::divscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::divscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::divscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::divscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::divscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class RDivScalar : public TF
    {
    public:
        RDivScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "rdivscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=scalar/T1";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<RDivScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if(!checktensors({this->args[1].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::rdivscalar<Author, double>(this->getvar<double>(0, mem), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::rdivscalar<Author, float>(this->getvar<float>(0, mem), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::rdivscalar<Author, half>(this->getvar<half>(0, mem), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::rdivscalar<Author, nv_bfloat16>(this->getvar<nv_bfloat16>(0, mem), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::rdivscalar<Author, int32_t>(this->getvar<int32_t>(0, mem), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::rdivscalar<Author, int32_t>(this->getvar<int32_t>(0, mem), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::rdivscalar<Author, int16_t>(this->getvar<int16_t>(0, mem), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::rdivscalar<Author, int8_t>(this->getvar<int8_t>(0, mem), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // invert
    template <typename Author>
    class Invert : public TF
    {
    public:
        Invert(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "invert";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=~T1";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Invert<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if(!checktensors({this->args[0].textvalue,this->returns[0].textvalue},mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Int64:
                tensorfunc::invert<Author>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::invert<Author>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::invert<Author>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::invert<Author>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
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
