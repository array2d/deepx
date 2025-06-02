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

    // reshape
    template <typename Author>
    class Reshape : public TF
    {
    public:
        Reshape(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "reshape";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
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
            vector<int> shape = this->getvector<int>(1, true);
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

    // transpose
    template <typename Author>
    class Transpose : public TF
    {
    public:
        Transpose(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "transpose";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
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
            vector<int> dim_order = this->getvector<int>(1, true);
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

    // concat
    template <typename Author>
    class Concat : public TF
    {
    public:
        Concat(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "concat";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
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
            if (!checktensors({this->returns[0].textvalue}, mem, error) != 0)
            {
                return 1;
            }

            vector<string> tensor_names = this->getvector<string>(0, true);
            if (!checktensors(tensor_names, mem, error) != 0)
            {
                return 1;
            }
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

    // broadcastTo
    template <typename Author>
    class BroadcastTo : public TF
    {
    public:
        BroadcastTo(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "broadcastTo";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T2 = T1.broadcastTo(new_shape=[4,3,2])";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<BroadcastTo<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> new_shape = this->getvector<int>(1, true);
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (input_type != output_type)
            {
                error = "Type mismatch: " + precision_str(input_type) + " != " + precision_str(output_type);
                return 1;
            }
            switch (input_type)
            {
            case Precision::Float64:
                broadcastTo<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), new_shape, *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                broadcastTo<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), new_shape, *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                broadcastTo<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), new_shape, *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                broadcastTo<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), new_shape, *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                broadcastTo<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), new_shape, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                broadcastTo<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), new_shape, *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                broadcastTo<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), new_shape, *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:
                broadcastTo<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), new_shape, *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }
            return 0;
        }
    };

    // indexselect
    template <typename Author>
    class IndexSelect : public TF
    {
    public:
        IndexSelect(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "indexselect";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T2 = T1.indexselect(index=[1,2], axis=1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<IndexSelect<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;

            int axis = this->getvar<int>(2, mem, true);
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (input_type != output_type)
            {
                error = "output_type " + precision_str(output_type) + " or input_type " + precision_str(input_type) + " must be the same";
                return 1;
            }
            Precision index_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            if (index_type != Precision::Int64 && index_type != Precision::Int32)
            {
                error = "index_type " + precision_str(index_type) + " only support " + precision_str(Precision::Int64) + " or " + precision_str(Precision::Int32);
                return 1;
            }

            switch (input_type)
            {
            case Precision::Float64:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, double, int64_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<double>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, double, int32_t>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<double>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Float32:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, float, int64_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<float>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, float, int32_t>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<float>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Float16:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, half, int64_t>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<half>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, half, int32_t>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<half>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::BFloat16:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, nv_bfloat16, int64_t>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, nv_bfloat16, int32_t>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Int64:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, int64_t, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, int64_t, int32_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Int32:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, int32_t, int64_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<int32_t>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, int32_t, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<int32_t>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Int16:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, int16_t, int64_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<int16_t>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, int16_t, int32_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<int16_t>(this->returns[0].textvalue));
                }
                break;
            }
            case Precision::Int8:
            {
                if (index_type == Precision::Int64)
                {
                    indexselect<Author, int8_t, int64_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int64_t>(this->args[1].textvalue), axis, *mem->gettensor<int8_t>(this->returns[0].textvalue));
                }
                else if (index_type == Precision::Int32)
                {
                    indexselect<Author, int8_t, int32_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), *mem->gettensor<int32_t>(this->args[1].textvalue), axis, *mem->gettensor<int8_t>(this->returns[0].textvalue));
                }
                break;
            }
            default:
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }
            return 0;
        }
    };

    //repeat
    template <typename Author>
    class Repeat : public TF
    {
    public:
        Repeat(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "repeat";
            this->metadata.author = Author::name();
            this->tftype = "changeshape";
            this->args = args;
            this->returns = returns;
        }

        string math_formula() const override
        {
            return "T2 = T1.repeat(repeats=[3 4 5])";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Repeat<Author>>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            Precision input_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            vector<int> repeats = this->getvector<int>(1);
            Precision output_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (input_type != output_type)
            {
                error = "Type mismatch: " + precision_str(input_type) + " != " + precision_str(output_type);
                return 1;
            }
            switch (input_type)
            {
            case Precision::Float64:
                repeat<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), repeats, *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                repeat<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), repeats,  *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                repeat<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), repeats,  *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                repeat<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), repeats,  *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            case Precision::Int64:
                repeat<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), repeats, *mem->gettensor<int64_t>(this->returns[0].textvalue));
                break;
            case Precision::Int32:
                repeat<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), repeats,  *mem->gettensor<int32_t>(this->returns[0].textvalue));
                break;
            case Precision::Int16:
                repeat<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), repeats,  *mem->gettensor<int16_t>(this->returns[0].textvalue));
                break;
            case Precision::Int8:   
                repeat<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), repeats,   *mem->gettensor<int8_t>(this->returns[0].textvalue));
                break;
            case Precision::Bool:
                repeat<Author, bool>(*mem->gettensor<bool>(this->args[0].textvalue), repeats,  *mem->gettensor<bool>(this->returns[0].textvalue));
                break;
            default:    
                error = "Unsupported type: " + precision_str(input_type);
                return 1;
            }
            return 0;
        }
    };
};
#endif // DEEPX_TF_CHANGESHAPE_HPP
