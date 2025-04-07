#ifndef DEEPX_TF_CHANGESHAPE_HPP
#define DEEPX_TF_CHANGESHAPE_HPP

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#include "deepx/tensorfunc/changeshape_miaobyte.hpp"

namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    using namespace std;
    template <typename Author>
    class Reshape : public TF
    {
    public:
        Reshape(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "reshape";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T1.reshape(shape)->T2";
        }

        shared_ptr<TF> clone() const override
        {
            return make_shared<Reshape<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> shape = this->getvector<int>(1, -1);
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (input_type != output_type)
            {
                error = "Type mismatch: " + precision_str(input_type) + " != " + precision_str(output_type);
                return 1;
            }
            switch (input_type)
            {
            case Precision::Float64:
                reshape<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), shape, *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                reshape<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), shape, *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                reshape<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), shape, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                reshape<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), shape, *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                reshape<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), shape, *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                reshape<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), shape, *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Transpose : public TF
    {
    public:
        Transpose(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "transpose";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T2 = T1.transpose(dimorder=[1,0])";
        }

        shared_ptr<TF> clone() const override
        {
            return make_shared<Transpose<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> dim_order = this->getvector<int>(1, -1);
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            
            if (input_type != output_type)
            {
                error = "Type mismatch: " + precision_str(input_type) + " != " + precision_str(output_type);
                return 1;
            }
            switch (input_type)
            {
            case Precision::Float64:
                transpose<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), dim_order, *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                transpose<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), dim_order, *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                transpose<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), dim_order, *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                transpose<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), dim_order, *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                transpose<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), dim_order, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                transpose<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), dim_order, *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                transpose<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), dim_order, *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                transpose<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), dim_order, *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }
            return 0;
        }
    };

        template <typename Author>
    class Concat : public TF
    {
    public:
        Concat(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "concat";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "Tresult = concat([T1, T2...], axis=3)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Concat>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            vector<string> tensor_names = this->getvector<string>(0, true);
            Precision input_type = mem->gettensor(tensor_names[0]).get()->shape.dtype;
            int axis = this->getvar<int>(1, mem, true);
            switch (input_type)
            {
            case Precision::Float64:
            {
                std::vector<Tensor<double> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<double>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<double>(this->returns[0].textvalue).get();
                concat<Author, double>(input, axis, *output);
                break;
            }
            case Precision::Float32:
            {
                std::vector<Tensor<float> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<float>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<float>(this->returns[0].textvalue).get();
                concat<Author, float>(input, axis, *output);
                break;
            }
            case Precision::Float16:
            {
                std::vector<Tensor<half> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<half>(tensor_names[i]).get());   
                }
                auto output = mem->gettensor<half>(this->returns[0].textvalue).get();
                concat<Author, half>(input, axis, *output);
                break;
            }
            case Precision::BFloat16:
            {
                std::vector<Tensor<nv_bfloat16> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<nv_bfloat16>(tensor_names[i]).get());
                }   
                auto output = mem->gettensor<nv_bfloat16>(this->returns[0].textvalue).get();
                concat<Author, nv_bfloat16>(input, axis, *output);
                break;
            }
            case Precision::Int64:
            {
                std::vector<Tensor<int64_t> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<int64_t>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<int64_t>(this->returns[0].textvalue).get();
                concat<Author, int64_t>(input, axis, *output);
                break;
            }
            case Precision::Int32:
            {
                std::vector<Tensor<int32_t> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<int32_t>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<int32_t>(this->returns[0].textvalue).get();
                concat<Author, int32_t>(input, axis, *output);
                break;
            }
            case Precision::Int16:
            {
                std::vector<Tensor<int16_t> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<int16_t>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<int16_t>(this->returns[0].textvalue).get();
                concat<Author, int16_t>(input, axis, *output);
                break;
            }
            case Precision::Int8:
            {
                std::vector<Tensor<int8_t> *> input;
                for (int i = 0; i < tensor_names.size(); i++)
                {
                    input.push_back(mem->gettensor<int8_t>(tensor_names[i]).get());
                }
                auto output = mem->gettensor<int8_t>(this->returns[0].textvalue).get();
                concat<Author, int8_t>(input, axis, *output);
                break;
            }
            default:
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }

            return 0;
        };
    };

}
#endif // DEEPX_TF_CHANGESHAPE_HPP
