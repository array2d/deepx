#ifndef DEEPX_TF_ELEMENTWISE_HPP
#define DEEPX_TF_ELEMENTWISE_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include "deepx/dtype_ompsimd.hpp"
#include "deepx/mem/mem_ompsimd.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensorfunc/elementwise_cblas.hpp"
namespace deepx::tf
{
        // todtype
    class Todtype : public TF
    {
    public:
        Todtype(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "todtype";
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T3(dtypeA)->T1(dtypeB)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Todtype>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            auto a_shape = mem->gettensor(this->args[0].textvalue).get()->shape;
            auto c_shape = mem->gettensor(this->returns[0].textvalue).get()->shape;
            if (a_shape.size != c_shape.size)
            {
                error = "Shape mismatch: " +  to_string(a_shape.size)  + " != " +  to_string(c_shape.size);
                return 1;
            }
            Precision a_type = a_shape.dtype;
            Precision c_type = c_shape.dtype;
            switch (a_type)
            {
            case Precision::Float64:
            {
                switch (c_type)
                {
                case Precision::Float64:
                {
                    auto a = mem->gettensor<double>(this->args[0].textvalue);
                    auto b = mem->gettensor<double>(this->returns[0].textvalue);
                    b->copyer(a->data, b->data, a->shape.size);
                    break;
                }
                case Precision::Float32:
                    tensorfunc::todtype<double, float>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                    break;
                 case Precision::Int64:
                    tensorfunc::todtype<double, int64_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int32:
                    tensorfunc::todtype<double, int32_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int16:
                    tensorfunc::todtype<double, int16_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int8:
                    tensorfunc::todtype<double, int8_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                    break;
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
                break;
            }
            case Precision::Float32:
            {
                switch (c_type)
                {
                case Precision::Float64:
                    tensorfunc::todtype<float, double>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                    break;
                case Precision::Float32:
                {
                    auto a = mem->gettensor<float>(this->args[0].textvalue);
                    auto b = mem->gettensor<float>(this->returns[0].textvalue);
                    b->copyer(a->data,b->data, a->shape.size);
                    break;
                }
                case Precision::Int64:
                    tensorfunc::todtype<float, int64_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int32:
                    tensorfunc::todtype<float, int32_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int16:
                    tensorfunc::todtype<float, int16_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int8:
                    tensorfunc::todtype<float, int8_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                    break;
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
            }
 
            break;
            case Precision::Int64:
            {
                switch (c_type)
                {
                case Precision::Float64:
                    tensorfunc::todtype<int64_t, double>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                    break;
                case Precision::Float32:
                    tensorfunc::todtype<int64_t, float>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                    break;
                case Precision::Int64:
                {
                    auto a = mem->gettensor<int64_t>(this->args[0].textvalue);
                    auto b = mem->gettensor<int64_t>(this->returns[0].textvalue);
                    b->copyer(a->data, b->data, a->shape.size);
                    break;
                }
                case Precision::Int32:
                    tensorfunc::todtype<int64_t, int32_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int16:
                    tensorfunc::todtype<int64_t, int16_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int8:
                    tensorfunc::todtype<int64_t, int8_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                    break;
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
            }
            break;
            case Precision::Int32:
            {
                switch (c_type)
                {
                case Precision::Float64:
                    tensorfunc::todtype<int32_t, double>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                    break;
                case Precision::Float32:
                    tensorfunc::todtype<int32_t, float>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                    break;
                case Precision::Int64:
                    tensorfunc::todtype<int32_t, int64_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int32:
                {
                    auto a = mem->gettensor<int32_t>(this->args[0].textvalue);
                    auto b = mem->gettensor<int32_t>(this->returns[0].textvalue);
                    b->copyer(a->data, b->data, a->shape.size);
                    break;
                }
                case Precision::Int16:
                    tensorfunc::todtype<int32_t, int16_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int8:
                    tensorfunc::todtype<int32_t, int8_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                    break;
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
            }
            break;
            case Precision::Int16:
            {
                switch (c_type)
                {
                case Precision::Float64:
                    tensorfunc::todtype<int16_t, double>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                    break;
                case Precision::Float32:
                    tensorfunc::todtype<int16_t, float>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                    break;
                case Precision::Int64:
                    tensorfunc::todtype<int16_t, int64_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int32:
                    tensorfunc::todtype<int16_t, int32_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int16:
                {
                    auto a = mem->gettensor<int16_t>(this->args[0].textvalue);
                    auto b = mem->gettensor<int16_t>(this->returns[0].textvalue);
                    b->copyer(a->data, b->data, a->shape.size);
                    break;
                }
                case Precision::Int8:
                    tensorfunc::todtype<int16_t, int8_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                    break;
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
            }
            break;
            case Precision::Int8:
            {
                switch (c_type)
                {
                case Precision::Float64:
                    tensorfunc::todtype<int8_t, double>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                    break;
                case Precision::Float32:
                    tensorfunc::todtype<int8_t, float>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                    break;
               case Precision::Int64:
                    tensorfunc::todtype<int8_t, int64_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int32:
                    tensorfunc::todtype<int8_t, int32_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int16:
                    tensorfunc::todtype<int8_t, int16_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                    break;
                case Precision::Int8:
                {
                    auto a = mem->gettensor<int8_t>(this->args[0].textvalue);
                    auto b = mem->gettensor<int8_t>(this->returns[0].textvalue);
                    b->copyer(a->data, b->data, a->shape.size);
                    break;
                }
                default:
                    error = "Unsupported dtype: " + precision_str(c_type);
                    return 1;
                }
            }
            break;
            default:
                error = "Unsupported dtype: " + precision_str(c_type);
                return 1;
            }
            return 0;
        };
        
    };


    // add
    template <typename Author>
    class Add : public TF
    {
    public:
        Add(vector<Param> args, vector<Param> returns)
        {
            this->name = "add";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::add<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        AddScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "addscalar";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::addscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        Sub(vector<Param> args, vector<Param> returns)
        {
            this->name = "sub";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::sub<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        SubScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "subscalar";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue,   this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::subscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        Mul(vector<Param> args, vector<Param> returns)
        {   
            this->name = "mul";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::mul<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
            case Precision::Bool:
                //TODO 暂时用int8,来计算bool类型
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
        MulScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "mulscalar";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue,   this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::mulscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        Div(vector<Param> args, vector<Param> returns)
        {   
            this->name = "div";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue,   this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Int64:
                tensorfunc::div<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        DivScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "divscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=T1/scalar";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<DivScalar<Author>>(*this);
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
                tensorfunc::divscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::divscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::divscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1, mem), *mem->gettensor<int64_t>(this->returns[0].textvalue));
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
        RDivScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "rdivscalar";
            this->metadata.author = Author::name();
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
                tensorfunc::rdivscalar<Author, double>( this->getvar<double>(0, mem),*mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::rdivscalar<Author, float>(this->getvar<float>(0, mem),*mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::rdivscalar<Author, int64_t>(this->getvar<int64_t>(0, mem),*mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::rdivscalar<Author, int32_t>(this->getvar<int32_t>(0, mem),*mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::rdivscalar<Author, int16_t>(this->getvar<int16_t>(0, mem),*mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::rdivscalar<Author, int8_t>(this->getvar<int8_t>(0, mem),*mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
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
        Invert(vector<Param> args, vector<Param> returns)
        {
            this->name = "invert";
            this->metadata.author = Author::name();
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
            if (!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error)!=0)
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
            case Precision::Bool:
                tensorfunc::invert<Author>(*mem->gettensor<bool>(this->args[0].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Sqrt : public TF
    {
    public:
        Sqrt(vector<Param> args, vector<Param> returns)
        {
            this->name = "sqrt";
            this->metadata.author = Author::name();  
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;   
            }
            return 0;
        }
    };

    template <typename Author>
    class Pow : public TF
    {
    public:
        Pow(vector<Param> args, vector<Param> returns)
        {
            this->name = "pow";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override    
        {
            return "T3=T1^T2";
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
            if (a_type != b_type || a_type != c_type)   
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " != " + precision_str(c_type);
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };  

    template <typename Author>
    class PowScalar : public TF
    {
    public:
        PowScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "powscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=T1^scalar";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<PowScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
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
                tensorfunc::powscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::powscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // rpowscalar
    template <typename Author>
    class RpowScalar : public TF
    {
    public:
        RpowScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "rpowscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=scalar^T1";
        }
        shared_ptr<TF> clone() const override
        {   
            return make_shared<RpowScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
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
                tensorfunc::rpowscalar<Author, double>(this->getvar<double>(0, mem), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:    
                tensorfunc::rpowscalar<Author, float>(this->getvar<float>(0, mem), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }   
            return 0;
        }
    };

    template <typename Author>
    class Log : public TF
    {
    public:
        Log(vector<Param> args, vector<Param> returns)
        {
            this->name = "log";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Exp : public TF
    {
    public:
        Exp(vector<Param> args, vector<Param> returns)
        {   
            this->name = "exp";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
    
    template <typename Author>
    class Sin : public TF
    {
    public:
        Sin(vector<Param> args, vector<Param> returns)
        {
            this->name = "sin";
            this->metadata.author = Author::name();
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Cos : public TF
    {
    public:
        Cos(vector<Param> args, vector<Param> returns)
        {   
            this->name = "cos";
            this->metadata.author = Author::name();
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
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
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
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
    
    template <typename Author>
    class Tan : public TF
    {
    public:
        Tan(vector<Param> args, vector<Param> returns)
        {   
            this->name = "tan";
            this->metadata.author = Author::name();
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
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
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
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };  
    
    template <typename Author>
    class Max : public TF
    {
    public:
        Max(vector<Param> args, vector<Param> returns)
        {
            this->name = "max";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=max(T1,T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Max<Author>>(*this);
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
                tensorfunc::max<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::max<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
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
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
    
    template <typename Author>
    class MaxScalar : public TF
    {
    public:
        MaxScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "maxscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=max(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<MaxScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if ( a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type)  + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::maxscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::maxscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::maxscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::maxscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::maxscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::maxscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Min : public TF
    {
    public:
        Min(vector<Param> args, vector<Param> returns)
        {   
            this->name = "min"; 
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=min(T1,T2)";
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
            if (a_type != b_type || a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " != " + precision_str(c_type);
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
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class MinScalar : public TF
    {
    public:
        MinScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "minscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=min(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<MinScalar<Author>>(*this);
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
                tensorfunc::minscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::minscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::minscalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::minscalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::minscalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::minscalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    //equal
    template <typename Author>
    class Equal : public TF
    {
    public:
        Equal(vector<Param> args, vector<Param> returns)
        {   
            this->name = "equal";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "equal(T1,T2)->mask";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Equal<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            float epsilon =0;
            if (this->args.size()>2){
                epsilon=this->getvar<float>(2,mem,true);
            }
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || mask_type!=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + "  " + precision_str(mask_type)+"!=bool";
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::equal<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            case Precision::Float32:
                tensorfunc::equal<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::equal<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::equal<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::equal<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::equal<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }   
    };


    template <typename Author>
    class EqualScalar : public TF
    {
    public:
        EqualScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "equalscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=equal(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<EqualScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;    
            float epsilon = this->getvar<float>(2,mem,true);
            if (mask_type !=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(mask_type)+"!=bool";
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::equalscalar<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::equalscalar<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::equalscalar<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::equalscalar<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::equalscalar<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::equalscalar<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    //notequal
    template <typename Author>
    class NotEqual : public TF
    {
    public:
        NotEqual(vector<Param> args, vector<Param> returns)
        {   
            this->name = "notequal";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "notequal(T1,T2)->mask";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<NotEqual<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            float epsilon =0;
            if (this->args.size()>2){
                epsilon=this->getvar<float>(2,mem,true);
            }
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != b_type || mask_type!=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + "  " + precision_str(mask_type)+"!=bool";
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::notequal<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            case Precision::Float32:
                tensorfunc::notequal<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::notequal<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::notequal<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::notequal<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::notequal<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }   
    };


    template <typename Author>
    class NotEqualScalar : public TF
    {
    public:
        NotEqualScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "notequalscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=notequal(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<NotEqualScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;    
            float epsilon = this->getvar<float>(2,mem,true);
            if (mask_type !=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(mask_type)+"!=bool";
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::notequalscalar<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::notequalscalar<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::notequalscalar<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::notequalscalar<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::notequalscalar<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::notequalscalar<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), epsilon, *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    //less
    template <typename Author>
    class Less : public TF
    {
    public:
        Less(vector<Param> args, vector<Param> returns)
        {
            this->name = "less";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=less(T1,T2)";
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
            if (a_type != b_type || mask_type!=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + "  " + precision_str(mask_type)+"!=bool";
                return 1;    
            }   
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::less<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::less<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::less<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::less<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::less<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;      
            case Precision::Int8:
                tensorfunc::less<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }   
            return 0;
        }
    };

    //lessscalar
    template <typename Author>
    class LessScalar : public TF
    {
    public:
        LessScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "lessscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {   
            return "mask=less(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<LessScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override    
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (mask_type!=Precision::Bool)
            {
                error = "mask_type should be "  +precision_str(Precision::Bool);
                return 1;
            }
            switch (a_type)
            {   
            case Precision::Float64:
                tensorfunc::lessscalar<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::lessscalar<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;      
            case Precision::Int64:
                tensorfunc::lessscalar<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::lessscalar<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            case Precision::Int16:
                tensorfunc::lessscalar<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;  
            case Precision::Int8:
                tensorfunc::lessscalar<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;    
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }       
            return 0;
        }   
    };  
    
    //greater
    template <typename Author>
    class Greater : public TF
    {
    public:
        Greater(vector<Param> args, vector<Param> returns)
        {
            this->name = "greater";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=greater(T1,T2)";
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
            if (a_type != b_type || mask_type!=Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + "  " + precision_str(mask_type)+"!=bool";
                return 1;
            }
            switch (a_type) 
            {
            case Precision::Float64:
                tensorfunc::greater<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::greater<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::greater<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::greater<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::greater<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue)); 
                break;
            case Precision::Int8:
                tensorfunc::greater<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;   
            }
            return 0;
        }
    };

    //greaterscalar
    template <typename Author>
    class GreaterScalar : public TF
    {
    public:
        GreaterScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "greaterscalar";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=greater(T1,scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<GreaterScalar<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision mask_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (mask_type!=Precision::Bool)
            {
                error = "mask_type should be "  +precision_str(Precision::Bool);
                return 1;
            }   
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::greaterscalar<Author, double,bool>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::greaterscalar<Author, float,bool>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::greaterscalar<Author, int64_t,bool>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::greaterscalar<Author, int32_t,bool>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::greaterscalar<Author, int16_t,bool>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::greaterscalar<Author, int8_t,bool>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;   
            }    
            return 0;
        }   
    };  

    //switch
    template <typename Author>
    class Switch : public TF
    {
    public:
        Switch(vector<Param> args, vector<Param> returns)
        {
            this->name = "switch";
            this->metadata.author = Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "C=switch([tensors],case)";
        }
        shared_ptr<TF> clone() const override   
        {
            return make_shared<Switch<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision cases_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (cases_type != Precision::Int32 && cases_type != Precision::Bool)
            {
                error = "Type mismatch: " + precision_str(cases_type) + " != Int32 or Bool";
                return 1;
            }

            switch (output_type)
            {
            case Precision::Float64:
                if (cases_type == Precision::Bool)  
                {
                    tensorfunc::Switch<Author, double,bool>(mem->gettensors<double>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, double,int32_t>(mem->gettensors<double>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                }
                break;
            case Precision::Float32:
                if (cases_type == Precision::Bool)      
                {
                    tensorfunc::Switch<Author, float,bool>(mem->gettensors<float>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, float,int32_t>(mem->gettensors<float>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                }
                break;
            case Precision::Int64:
                if (cases_type == Precision::Bool)      
                {
                    tensorfunc::Switch<Author, int64_t,bool>(mem->gettensors<int64_t>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, int64_t,int32_t>(mem->gettensors<int64_t>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int64_t>(this->returns[0].textvalue));
                }
                break;
            case Precision::Int32:
                if (cases_type == Precision::Bool)          
                {
                    tensorfunc::Switch<Author, int32_t,bool>(mem->gettensors<int32_t>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, int32_t,int32_t>(mem->gettensors<int32_t>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int32_t>(this->returns[0].textvalue));
                }
                break;
            case Precision::Int16:  
                if (cases_type == Precision::Bool)  
                {
                    tensorfunc::Switch<Author, int16_t,bool>(mem->gettensors<int16_t>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, int16_t,int32_t>(mem->gettensors<int16_t>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int16_t>(this->returns[0].textvalue));
                }
                break;
            case Precision::Int8:   
                if (cases_type == Precision::Bool) 
                {
                    tensorfunc::Switch<Author, int8_t,bool>(mem->gettensors<int8_t>(this->getvector<string>(0)), *mem->gettensor<bool>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                }
                else
                {
                    tensorfunc::Switch<Author, int8_t,int32_t>(mem->gettensors<int8_t>(this->getvector<string>(0)), *mem->gettensor<int32_t>(this->args[1].textvalue), *mem->gettensor<int8_t>(this->returns[0].textvalue));
                }
                break;
            default:
                error = "Unsupported dtype: " + precision_str(cases_type);
                return 1;
            }
            return 0;
        }
    };

 
};
#endif // DEEPX_TF_ELEMENTWISE_HPP
