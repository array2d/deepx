#ifndef DEEPX_TF_ELEMENTWISE_SIN_HPP
#define DEEPX_TF_ELEMENTWISE_SIN_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tensorfunc/elementwise_miaobyte_sin.hpp"

namespace deepx::tf
{

    template <typename Author>
    class Sin : public TF
    {
    public:
        Sin(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "sin";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=sin(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Sin<Author>>(*this);
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
                tensorfunc::sin<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::sin<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::sin<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::sin<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Cos : public TF
    {
    public:
        Cos(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "cos";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=cos(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Cos<Author>>(*this);
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
                tensorfunc::cos<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::cos<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::cos<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::cos<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Tan : public TF
    {
    public:
        Tan(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "tan";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=tan(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Tan<Author>>(*this);
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
                tensorfunc::tan<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::tan<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
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
