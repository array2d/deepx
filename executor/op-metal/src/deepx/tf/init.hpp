#ifndef DEEPX_TF_INIT_HPP
#define DEEPX_TF_INIT_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"

namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    using namespace std;

    // ═══════════════════════════════════════════════════════════
    // Init ops: fill existing tensors with values.
    // These do NOT create tensors (that's heap-plat's job) —
    // they only modify values of already-allocated tensors.
    // ═══════════════════════════════════════════════════════════

    // constant — fill tensor with a scalar value
    template <typename Author>
    class Constant : public TF
    {
    public:
        Constant(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "constant";
            this->metadata.author = Author::name();
            this->tftype = "init";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "constant(value)->T1"; }
        shared_ptr<TF> clone() const override { return make_shared<Constant<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->returns[0].textvalue;
            auto tensor = mem->gettensor(name).get();
            if (tensor == nullptr) {
                error = "tensor not found: " + name;
                return 1;
            }
            auto type = tensor->shape.dtype;
            switch (type) {
            case Precision::Float64:
                tensorfunc::constant<Author, double>(*mem->gettensor<double>(name).get(), this->getvar<double>(0, mem));
                break;
            case Precision::Float32:
                tensorfunc::constant<Author, float>(*mem->gettensor<float>(name).get(), this->getvar<float>(0, mem));
                break;
            case Precision::Int64:
                tensorfunc::constant<Author, int64_t>(*mem->gettensor<int64_t>(name).get(), this->getvar<int64_t>(0, mem));
                break;
            case Precision::Int32:
                tensorfunc::constant<Author, int32_t>(*mem->gettensor<int32_t>(name).get(), this->getvar<int32_t>(0, mem));
                break;
            case Precision::Int16:
                tensorfunc::constant<Author, int16_t>(*mem->gettensor<int16_t>(name).get(), this->getvar<int16_t>(0, mem));
                break;
            case Precision::Int8:
                tensorfunc::constant<Author, int8_t>(*mem->gettensor<int8_t>(name).get(), this->getvar<int8_t>(0, mem));
                break;
            case Precision::Bool:
                tensorfunc::constant<Author, bool>(*mem->gettensor<bool>(name).get(), this->getvar<bool>(0, mem));
                break;
            default:
                error = "unsupported dtype: " + precision_str(type);
                return 1;
            }
            return 0;
        }
    };

    // arange — fill with [start, start+step, start+2*step, ...]
    template <typename Author>
    class Arange : public TF
    {
    public:
        Arange(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "arange";
            this->metadata.author = Author::name();
            this->tftype = "init";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "arange(start,step)->T1"; }
        shared_ptr<TF> clone() const override { return make_shared<Arange<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->returns[0].textvalue;
            auto tensor = mem->gettensor(name).get();
            auto type = tensor->shape.dtype;
            switch (type) {
            case Precision::Float64: {
                auto output = mem->gettensor<double>(name).get();
                tensorfunc::arange<Author, double>(*output, this->getvar<double>(0, mem), this->getvar<double>(1, mem));
                break;
            }
            case Precision::Float32: {
                auto output = mem->gettensor<float>(name).get();
                tensorfunc::arange<Author, float>(*output, this->getvar<float>(0, mem), this->getvar<float>(1, mem));
                break;
            }
            case Precision::Int64: {
                auto output = mem->gettensor<int64_t>(name).get();
                tensorfunc::arange<Author, int64_t>(*output, this->getvar<int64_t>(0, mem), this->getvar<int64_t>(1, mem));
                break;
            }
            case Precision::Int32: {
                auto output = mem->gettensor<int32_t>(name).get();
                tensorfunc::arange<Author, int32_t>(*output, this->getvar<int32_t>(0, mem), this->getvar<int32_t>(1, mem));
                break;
            }
            case Precision::Int16: {
                auto output = mem->gettensor<int16_t>(name).get();
                tensorfunc::arange<Author, int16_t>(*output, this->getvar<int16_t>(0, mem), this->getvar<int16_t>(1, mem));
                break;
            }
            case Precision::Int8: {
                auto output = mem->gettensor<int8_t>(name).get();
                tensorfunc::arange<Author, int8_t>(*output, this->getvar<int8_t>(0, mem), this->getvar<int8_t>(1, mem));
                break;
            }
            default:
                error = "unsupported dtype: " + precision_str(type);
                return 1;
            }
            return 0;
        }
    };

} // namespace deepx::tf

#endif // DEEPX_TF_INIT_HPP
