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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
                tensorfunc::minscalar<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
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
    class Equal : public TF
    {
    public:
        Equal(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "equal";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Equal<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            float epsilon = this->getvar<float>(2, mem);
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " or " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::equal<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::equal<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::equal<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::equal<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::equal<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::equal<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::equal<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::equal<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class EqualScalar : public TF
    {
    public:
        EqualScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "equalscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<EqualScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            float epsilon = this->getvar<float>(2, mem);
            if (a_type != mask_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::equalscalar<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::equalscalar<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::equalscalar<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::equalscalar<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::equalscalar<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::equalscalar<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::equalscalar<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::equalscalar<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // less
    template <typename Author>
    class Less : public TF
    {
    public:
        Less(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "less";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Less<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " or " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::less<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::less<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::less<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::less<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::less<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::less<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::less<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::less<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // lessscalar
    template <typename Author>
    class LessScalar : public TF
    {
    public:
        LessScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "lessscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<LessScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != mask_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::lessscalar<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::lessscalar<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::lessscalar<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::lessscalar<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::lessscalar<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::lessscalar<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::lessscalar<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::lessscalar<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // greater
    template <typename Author>
    class Greater : public TF
    {
    public:
        Greater(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "greater";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Greater<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " or " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::greater<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::greater<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::greater<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::greater<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::greater<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::greater<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::greater<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::greater<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // greaterscalar
    template <typename Author>
    class GreaterScalar : public TF
    {
    public:
        GreaterScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "greaterscalar";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "mask=compare(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<GreaterScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != mask_type || mask_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(mask_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::greaterscalar<Author, double, bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::greaterscalar<Author, float, bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::greaterscalar<Author, half, bool>(*mem->gettensor<half>(this->args[0].textvalue), this->getvar<half>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::greaterscalar<Author, nv_bfloat16, bool>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), this->getvar<nv_bfloat16>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::greaterscalar<Author, int64_t, bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::greaterscalar<Author, int32_t, bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::greaterscalar<Author, int16_t, bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::greaterscalar<Author, int8_t, bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1, mem), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // switch
    template <typename Author>
    class Switch : public TF
    {
    public:
        Switch(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "switch";
            this->author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "C=switch(tensors,cases)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Switch<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {

            Precision C_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;

            switch (C_type)
            {
            case Precision::Float64:
                tensorfunc::Switch<Author, double>(mem->gettensors<double>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::Switch<Author, float>(mem->gettensors<float>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::Switch<Author, half>(mem->gettensors<half>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::Switch<Author, nv_bfloat16>(mem->gettensors<nv_bfloat16>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::Switch<Author, int64_t>(mem->gettensors<int64_t>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::Switch<Author, int32_t>(mem->gettensors<int32_t>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::Switch<Author, int16_t>(mem->gettensors<int16_t>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::Switch<Author, int8_t>(mem->gettensors<int8_t>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            case Precision::Bool:
                tensorfunc::Switch<Author, bool>(mem->gettensors<bool>(this->getvector<string>(0)), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(C_type);
                return 1;
            }
            return 0;
        }
    };

};
#endif // DEEPX_TF_ELEMENTWISE_COMPARE_HPP
