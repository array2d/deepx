#ifndef DEEPX_TF_REDUCE_HPP
#define DEEPX_TF_REDUCE_HPP

#include <vector>
#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/reduce_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    using namespace std;

    template <typename Author>
    class Sum : public TF
    {
    public:
        Sum(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "sum";
            this->metadata.author = Author::name();
            this->tftype = "reduce";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "B = sum(A, dims, keepdims)"; }
        shared_ptr<TF> clone() const override { return make_shared<Sum>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> dims = this->getvector<int>(1, true);
            bool keepdims = this->getvar<bool>(2, mem, true);
            switch (input_type) {
            case Precision::Float64: sum<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: sum<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   sum<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   sum<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   sum<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    sum<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported type: " + precision_str(input_type); return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Prod : public TF
    {
    public:
        Prod(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "prod";
            this->metadata.author = Author::name();
            this->tftype = "reduce";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "B = prod(A, dims, keepdims)"; }
        shared_ptr<TF> clone() const override { return make_shared<Prod>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> dims = this->getvector<int>(1, true);
            bool keepdims = this->getvar<bool>(2, mem, true);
            switch (input_type) {
            case Precision::Float64: prod<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: prod<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   prod<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   prod<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   prod<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    prod<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported type: " + precision_str(input_type); return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class ReduceMax : public TF
    {
    public:
        ReduceMax(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "reducemax";
            this->metadata.author = Author::name();
            this->tftype = "reduce";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "B = reducemax(A, dims, keepdims)"; }
        shared_ptr<TF> clone() const override { return make_shared<ReduceMax>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> dims = this->getvector<int>(1, true);
            bool keepdims = this->getvar<bool>(2, mem, true);
            switch (input_type) {
            case Precision::Float64: reducemax<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: reducemax<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   reducemax<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   reducemax<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   reducemax<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    reducemax<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported type: " + precision_str(input_type); return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class ReduceMin : public TF
    {
    public:
        ReduceMin(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "reducemin";
            this->metadata.author = Author::name();
            this->tftype = "reduce";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "B = reducemin(A, dims, keepdims)"; }
        shared_ptr<TF> clone() const override { return make_shared<ReduceMin>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> dims = this->getvector<int>(1, true);
            bool keepdims = this->getvar<bool>(2, mem, true);
            switch (input_type) {
            case Precision::Float64: reducemin<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<double>(this->returns[0].textvalue)); break;
            case Precision::Float32: reducemin<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<float>(this->returns[0].textvalue)); break;
            case Precision::Int64:   reducemin<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int64_t>(this->returns[0].textvalue)); break;
            case Precision::Int32:   reducemin<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int32_t>(this->returns[0].textvalue)); break;
            case Precision::Int16:   reducemin<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int16_t>(this->returns[0].textvalue)); break;
            case Precision::Int8:    reducemin<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), dims, keepdims, *mem->gettensor<int8_t>(this->returns[0].textvalue)); break;
            default: error = "Unsupported type: " + precision_str(input_type); return 1;
            }
            return 0;
        }
    };

} // namespace deepx::tf

#endif // DEEPX_TF_REDUCE_HPP
