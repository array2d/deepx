#ifndef DEEPX_TF_TENSORLIFE_HPP
#define DEEPX_TF_TENSORLIFE_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "stdutil/num.hpp"

namespace deepx::tf
{
    class NewTensor : public TF
    {
    public:
        NewTensor(vector<Param> args, vector<Param> returns)
        {
            this->name = "newtensor";
            this->tftype = "tensorlife";
            this->args = args;
            this->returns = returns;
        }
 
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->returns[0].textvalue;
            TypeDef type = this->returns[0].dtype;
            if (uint8_t(type.category() & DataCategory::Tensor) == 0)
            {
                error = "newtensor: return type must include tensor category";
                return 1;
            }
            vector<int> shape=this->getvector<int>(0);
            switch (type.precision())
            {
            case Precision::Float32:
            {
                Tensor<float> t = tensorfunc::New<float>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Float64:
            {
                Tensor<double> t = tensorfunc::New<double>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Float16:
            {
                error = "newtensor: Float16 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Float8E5M2:
            {
                error = "newtensor: Float8E5M2 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Float8E4M3:
            {
                error = "newtensor: Float8E4M3 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Float4E2M1:
            {
                error = "newtensor: Float4E2M1 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Int64:
            {
                Tensor<int64_t> t = tensorfunc::New<int64_t>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Int32:
            {
                Tensor<int32_t> t = tensorfunc::New<int32_t>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Int16:
            {
                Tensor<int16_t> t = tensorfunc::New<int16_t>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Int8:
            {
                Tensor<int8_t> t = tensorfunc::New<int8_t>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::Int4:
            {
                error = "newtensor: Int4 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Bool:
            {
                Tensor<bool> t = tensorfunc::New<bool>(shape);
                mem->addtensor(name, t);
                break;
            }
            case Precision::String:
            {
                Tensor<string> t = tensorfunc::New<string>(shape);
                mem->addtensor(name, t);
                break;
            }
            default:
            {
                error = "newtensor: unsupported precision";
                return 1;
            }
            };
            return 0;
        };

        string math_formula() const override
        {
            return "T1 =Tensor(shape=[...])";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<NewTensor>(*this);
        }
    };

    class CopyTensor : public TF
    {
    public:
        CopyTensor(vector<Param> args, vector<Param> returns)
        {
            this->name = "copytensor";
            this->args = args;
            this->returns = returns;
            this->tftype = "tensorlife";
        }
 
         int run(shared_ptr<MemBase> mem, string &error) override
        {
            if (!checktensors({this->args[0].textvalue, this->args[1].textvalue}, mem, error) != 0)
            {
                return 1;
            }
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            if (input_type != type)
            {
                error = "copytensor: input type and return type must be the same";
                return 1;
            }
            switch (input_type)
            {
            case Precision::Float64:
            {
                tensorfunc::copy(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue));
                break;
            }
            case Precision::Float32:
            {
                tensorfunc::copy(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue));
                break;
            }
            case Precision::Int64:
            {
                tensorfunc::copy(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue));
                break;
            }
            case Precision::Int32:
            {
                tensorfunc::copy(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue));
                break;
            }
            case Precision::Int16:
            {
                tensorfunc::copy(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int16_t>(this->args[1].textvalue));
                break;
            }
            case Precision::Int8:
            {
                tensorfunc::copy(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int8_t>(this->args[1].textvalue));
                break;
            }
            case Precision::Bool:
            {
                tensorfunc::copy(*mem->gettensor<bool>(this->args[0].textvalue), *mem->gettensor<bool>(this->args[1].textvalue));
                break;
            }
            default:
            {
                error = "copytensor: unsupported precision";
                return 1;
            }
            };
            return 0;
        }

        string math_formula() const override
        {
            return "T2.data = T1.data";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<CopyTensor>(*this);
        }
    };

    class DelTensor : public TF
    {
    public:
        DelTensor(vector<Param> args, vector<Param> returns)
        {
            this->name = "deltensor";
            this->args = args;
            this->returns = returns;
            this->tftype = "tensorlife";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            mem->delete_tensor(name);
            return 0;
        }

        string math_formula() const override
        {
            return "del T1";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<DelTensor>(*this);
        }
    };

    //rename
    class RenameTensor : public TF
    {
    public:
        RenameTensor(vector<Param> args, vector<Param> returns)
        {
            this->name = "renametensor";
            this->args = args;
            this->returns = returns;
            this->tftype = "tensorlife";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string old_name = this->args[0].textvalue;
            if (!checktensors({this->args[0].textvalue}, mem, error) != 0)
            {
                return 1;
            }
            string new_name = this->args[1].textvalue;
            mem->rename_tensor(old_name, new_name);
            return 0;
        }
        string math_formula() const override
        {
            return "rename T1 to T2";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<RenameTensor>(*this);
        }
    };
}
#endif
