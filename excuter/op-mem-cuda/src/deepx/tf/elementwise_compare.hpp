#ifndef DEEPX_TF_ELEMENTWISE_COMPARE_HPP
#define DEEPX_TF_ELEMENTWISE_COMPARE_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tensorfunc/elementwise_miaobyte_compare.hpp"

namespace deepx::tf
{

    template <typename Author>
    class Max : public TF
    {
    public:
        Max(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "max";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Max(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "max")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=max(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Max<Author>>(*this);
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
                tensorfunc::max<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::max<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::max<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::max<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::max<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::max<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::max<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::max<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class MaxScalar : public TF
    {
    public:
        MaxScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "maxscalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        MaxScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "maxscalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=max(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<MaxScalar<Author>>(*this);
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
                tensorfunc::maxscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::maxscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::maxscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::maxscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::maxscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::maxscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;  
            case Precision::Int16:  
                tensorfunc::maxscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::maxscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Min : public TF
    {
    public:
        Min(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "min";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Min(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "min")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=min(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Min<Author>>(*this);
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
                tensorfunc::min<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::min<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;      
            case Precision::Float16:
                tensorfunc::min<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::min<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;  
            case Precision::Int64:  
                tensorfunc::min<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::min<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::min<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::min<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class MinScalar : public TF
    {
    public:
        MinScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "minscalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        MinScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "minscalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "T3=min(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<MinScalar<Author>>(*this);
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
                tensorfunc::minscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::minscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::minscalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::minscalar   <Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::minscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::minscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::minscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::minscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Compare : public TF
    {
    public:
        Compare(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "compare";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        Compare(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "compare")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "mask=compare(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Compare<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::compare<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::compare<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::compare<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::compare<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::compare<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::compare<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;  
            case Precision::Int16:
                tensorfunc::compare<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::compare<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class CompareScalar : public TF
    {
    public:
        CompareScalar(const vector<Param> &args, const vector<Param> &returns)
        {   
            this->name = "comparescalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        CompareScalar(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "comparescalar")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        string math_formula() const override
        {
            return "mask=compare(T1, scalar)";  
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<CompareScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != mask_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::comparescalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::comparescalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::comparescalar<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::comparescalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::comparescalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::comparescalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::comparescalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::comparescalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
        
};
#endif // DEEPX_TF_ELEMENTWISE_COMPARE_HPP
