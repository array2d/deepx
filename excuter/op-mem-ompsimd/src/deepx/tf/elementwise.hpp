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

    template <typename Author>
    class Add : public TF
    {
    public:
        Add(vector<Param> args, vector<Param> returns)
        {
            this->name = "add";
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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

    template <typename Author>
    class Sqrt : public TF
    {
    public:
        Sqrt(vector<Param> args, vector<Param> returns)
        {
            this->name = "sqrt";
            this->author = Author::name();  
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
            this->author = Author::name();
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
            this->author = Author::name();
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

    template <typename Author>
    class Log : public TF
    {
    public:
        Log(vector<Param> args, vector<Param> returns)
        {
            this->name = "log";
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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
            this->author = Author::name();
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

    template <typename Author>
    class Compare : public TF
    {
    public:
        Compare(vector<Param> args, vector<Param> returns)
        {   
            this->name = "compare";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=compare(T1,T2)";
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
            if (a_type != b_type || a_type != mask_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(b_type) + " != " + precision_str(mask_type);
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
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }   
    };


    template <typename Author>
    class CompareScalar : public TF
    {
    public:
        CompareScalar(vector<Param> args, vector<Param> returns)
        {
            this->name = "comparescalar";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "mask=compare(T1,scalar)";
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
                tensorfunc::comparescalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::comparescalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                tensorfunc::comparescalar<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), this->getvar<int64_t>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                tensorfunc::comparescalar<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), this->getvar<int32_t>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                tensorfunc::comparescalar<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), this->getvar<int16_t>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                tensorfunc::comparescalar<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), this->getvar<int8_t>(1,mem,true), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported dtype: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
};

#endif
