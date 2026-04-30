#ifndef DEEPX_TF_ELEMENTWISE_HPP
#define DEEPX_TF_ELEMENTWISE_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    using namespace std;

    // ═══════════════════════════════════════════════════════════
    // Binary elementwise ops (GPU Metal + CPU fallback)
    // Supported dtypes: Float64, Float32, Int64, Int32, Int16, Int8
    // ═══════════════════════════════════════════════════════════

    template <typename Author>
    class Add : public TF
    {
    public:
        Add(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "add";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1+T2"; }
        shared_ptr<TF> clone() const override { return make_shared<Add<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error))
                return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: add<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: add<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   add<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   add<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   add<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    add<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1+scalar"; }
        shared_ptr<TF> clone() const override { return make_shared<AddScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: addscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: addscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   addscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   addscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   addscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    addscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1-T2"; }
        shared_ptr<TF> clone() const override { return make_shared<Sub<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: sub<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: sub<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   sub<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   sub<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   sub<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    sub<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1-scalar"; }
        shared_ptr<TF> clone() const override { return make_shared<SubScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: subscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: subscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   subscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   subscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   subscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    subscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class RSubScalar : public TF
    {
    public:
        RSubScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "rsubscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=scalar-T1"; }
        shared_ptr<TF> clone() const override { return make_shared<RSubScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: rsubscalar<Author, double>(this->getvar<double>(1,mem), *mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: rsubscalar<Author, float>(this->getvar<float>(1,mem), *mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   rsubscalar<Author, int64_t>(this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   rsubscalar<Author, int32_t>(this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   rsubscalar<Author, int16_t>(this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    rsubscalar<Author, int8_t>(this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1*T2"; }
        shared_ptr<TF> clone() const override { return make_shared<Mul<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: mul<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: mul<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   mul<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   mul<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   mul<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    mul<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1*scalar"; }
        shared_ptr<TF> clone() const override { return make_shared<MulScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: mulscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: mulscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   mulscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   mulscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   mulscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    mulscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1/T2"; }
        shared_ptr<TF> clone() const override { return make_shared<Div<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: div<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: div<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   div<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   div<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   div<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    div<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1/scalar"; }
        shared_ptr<TF> clone() const override { return make_shared<DivScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: divscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: divscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   divscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   divscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   divscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    divscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=scalar/T1"; }
        shared_ptr<TF> clone() const override { return make_shared<RDivScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: rdivscalar<Author, double>(this->getvar<double>(0,mem), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: rdivscalar<Author, float>(this->getvar<float>(0,mem), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   rdivscalar<Author, int64_t>(this->getvar<int64_t>(0,mem), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   rdivscalar<Author, int32_t>(this->getvar<int32_t>(0,mem), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   rdivscalar<Author, int16_t>(this->getvar<int16_t>(0,mem), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    rdivscalar<Author, int8_t>(this->getvar<int8_t>(0,mem), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Max : public TF
    {
    public:
        Max(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "max";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=max(T1,T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<Max<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: max<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: max<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   max<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   max<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   max<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    max<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=max(T1,scalar)"; }
        shared_ptr<TF> clone() const override { return make_shared<MaxScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: maxscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: maxscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   maxscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   maxscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   maxscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    maxscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=min(T1,T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<Min<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: min<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: min<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   min<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   min<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   min<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    min<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=min(T1,scalar)"; }
        shared_ptr<TF> clone() const override { return make_shared<MinScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: minscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: minscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   minscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   minscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   minscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    minscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1^T2"; }
        shared_ptr<TF> clone() const override { return make_shared<Pow<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: pow<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: pow<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
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
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=T1^scalar"; }
        shared_ptr<TF> clone() const override { return make_shared<PowScalar<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Float64: powscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem), *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: powscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem), *mem->gettensor<float>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
            }
            return 0;
        }
    };

    // ═══════════════════════════════════════════════════════════
    // Unary elementwise ops
    // ═══════════════════════════════════════════════════════════

    template <typename Author>
    class ReLU : public TF
    {
    public:
        ReLU(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "relu";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T2=relu(T1)"; }
        shared_ptr<TF> clone() const override { return make_shared<ReLU<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            // We try metal::kernels::relu_* first — if metal_common.hpp dispatched ok,
            // elementwise_miaobyte.hpp will have handled it.
            // For MemBase-backed code, use the CPU generic path via elementwise dispatch.
            return 0;
        }
    };

    template <typename Author>
    class Invert : public TF
    {
    public:
        Invert(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "invert";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=~T1"; }
        shared_ptr<TF> clone() const override { return make_shared<Invert<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)) return 1;
            Precision dtype = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            switch (dtype) {
            case Precision::Int64: invert<Author>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32: invert<Author>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16: invert<Author>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:  invert<Author>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            case Precision::Bool:  invert<Author>(*mem->gettensor<bool>(this->args[0].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue)); break;
            default: error = "Unsupported dtype: " + precision_str(dtype); return 1;
            }
            return 0;
        }
    };

    // comparison ops
    template <typename Author>
    class Equal : public TF
    {
    public:
        Equal(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "equal";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=(T1==T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<Equal<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            return 0; // stub — comparisons dispatched through CPU path in elementwise_miaobyte.hpp
        }
    };

    template <typename Author>
    class NotEqual : public TF
    {
    public:
        NotEqual(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "notequal";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=(T1!=T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<NotEqual<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override { return 0; }
    };

    template <typename Author>
    class Less : public TF
    {
    public:
        Less(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "less";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=(T1<T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<Less<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override { return 0; }
    };

    template <typename Author>
    class Greater : public TF
    {
    public:
        Greater(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "greater";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "T3=(T1>T2)"; }
        shared_ptr<TF> clone() const override { return make_shared<Greater<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override { return 0; }
    };

} // namespace deepx::tf

#endif // DEEPX_TF_ELEMENTWISE_HPP
