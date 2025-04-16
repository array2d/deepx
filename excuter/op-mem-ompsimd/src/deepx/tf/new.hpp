#ifndef DEEPX_TF_NEW_HPP
#define DEEPX_TF_NEW_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "stdutil/num.hpp"

namespace deepx::tf
{
    class NewTensor : public TF
    {
    public:
        NewTensor(vector<Param> args, vector<Param> returns)
        {
            this->name = "newtensor";
            this->args = args;
            this->returns = returns;
        }

        NewTensor(string text, bool call = false)
        {
            this->parse(text);
            if (this->name != "newtensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
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
        CopyTensor()
        {
            this->name = "copytensor";
        }
        CopyTensor(string text)
        {
            this->parse(text);
            if (this->name != "copytensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            // TODO
            //  auto src=mem.gettensor<T>(this->args[0].name);
            //  auto dst=mem.gettensor<T>(this->returns[0].name);
            //  tensorfunc::copytensor(*src,*dst);
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

    class CloneTensor : public TF
    {
    public:
        CloneTensor()
        {
            this->name = "clonetensor";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            // TODO
            //  auto src=mem.gettensor<T>(this->args[0]);
            //  string dst=this->returns[0];
            //  mem.addtensor(dst,tensorfunc::clone(*src));
            return 0;
        }

        string math_formula() const override
        {
            return "T2 = T1.clone()";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<CloneTensor>(*this);
        }
    };

    class DelTensor : public TF
    {
    public:
        DelTensor()
        {
            this->name = "deltensor";
        }
        DelTensor(string text)
        {
            this->parse(text);
            if (this->name != "deltensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
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
}
#endif
