#ifndef DEEPX_TF_CHANGESHAPE_HPP
#define DEEPX_TF_CHANGESHAPE_HPP

#include <vector>
#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.hpp"
#include "deepx/dtype.hpp"

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
            return "T2=T1.reshape(shape)";
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
                reshape<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), shape);
                break;
            case Precision::Float32:
                reshape<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), shape);
                break;
            case Precision::Int64:
                reshape<Author, int64_t>(*mem->gettensor<int64_t>(this->args[0].textvalue), shape);
                break;
            case Precision::Int32:
                reshape<Author, int32_t>(*mem->gettensor<int32_t>(this->args[0].textvalue), shape);
                break;
            case Precision::Int16:
                reshape<Author, int16_t>(*mem->gettensor<int16_t>(this->args[0].textvalue), shape);
                break;
            case Precision::Int8:
                reshape<Author, int8_t>(*mem->gettensor<int8_t>(this->args[0].textvalue), shape);
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
            this->name = _name;
            this->author = Author::name(); 
            this->args = args;
            this->returns = returns;
        }
 

        string math_formula() const override
        {
            return "Tresult = concat([T1, T2...], axis=3)";
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            // TODO，去掉T
            //  std::vector<Tensor<T> *> input;
            //  for (int i = 0; i < this->args.size() - 1; i++)
            //  {
            //      input.push_back(mem.gettensor<T>(this->args[i].name).get());
            //  }
            //  auto output = mem.gettensor<T>(this->returns[0].name).get();
            //  int axis = this->getvar<int>(-1,mem,false);
            //  tensorfunc::concat(input, axis, *output);
            return 0;
        };
        shared_ptr<TF> clone() const override
        {
            return make_shared<Concat>(*this);
        }
    };

    // class Split : public TF
    // {
    // public:
    //     Split()
    //     {
    //         this->name="split";
    //     }
    //     Split(string text)
    //     {
    //         this->parse(text);
    //         if (this->name!="split"){
    //             throw std::runtime_error("Invalid name: "+this->name);
    //         }
    //     }
    //     void funcdef() override
    //     {
    //         this->parse("split(float32 T1,int32 3)->(float32 T2,T3)");
    //     }
    //     string math_formula() const override
    //     {
    //         return "T2,T3 = split(T1, axis=3)";
    //     }
    //     void run(mem::Mem &mem) override
    //     {
    //         std::vector<Tensor<T> *> input;
    //         for (int i = 0; i < this->args.size() - 1; i++)
    //         {
    //             input.push_back(mem.gettensor<T>(this->args[i]).get());
    //         }
    //         int axis = mem.getarg<int>(this->args.back());
    //         auto output = mem.gettensor<T>(this->returns[0]).get();
    //         tensorfunc::split(*output, axis, input);
    //     }
    // };

    // template <typename T>
    // class Transpose : public Op
    // {
    // public:
    //     Transpose()
    //     {
    //         this->init("transpose", "any", {}, {}, false, {}, {});
    //     }
    //     Transpose(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {})
    //     {
    //         this->init("transpose", "any", args, returns, require_grad, args_grad, returns_grad);
    //     }
    //     Transpose(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {})
    //     {
    //         this->init("transpose", "any", args, returns, require_grad, args_grad, returns_grad);
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto input = mem.gettensor<T>(this->args[0]).get();
    //         vector<int> dimOrder;
    //         if (this->args.size() == 2 && !is_integer(this->args[1]))
    //         {
    //             dimOrder = mem.getvector<int32_t>(this->args[1]);
    //         }
    //         else if (this->args.size() > 2)
    //         {
    //             for (int i = 1; i < this->args.size(); i++)
    //             {
    //                 dimOrder.push_back(atoi(this->args[i].c_str()));
    //             }
    //         }
    //         auto output = mem.gettensor<T>(this->returns[0]).get();
    //         tensorfunc::transpose(*input, *output, dimOrder);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto input_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         vector<int> dimOrder;
    //         if (this->args.size() == 2 && !is_integer(this->args[1]))
    //         {
    //             dimOrder = mem.getvector<int32_t>(this->args[1]);
    //         }
    //         else if (this->args.size() > 2)
    //         {
    //             for (int i = 1; i < this->args.size(); i++)
    //             {
    //                 dimOrder.push_back(atoi(this->args[i].c_str()));
    //             }
    //         }
    //         auto output_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         tensorfunc::transpose(*output_grad, *input_grad, dimOrder);
    //     }
    //     void funcdef() override
    //     {
    //         this->init("transpose", "float32", {"T1", "1", "0"}, {"T2"}, false, {}, {});
    //     }
    //     string math_formula() const override
    //     {
    //         return "T2 = transpose(T1, dimorder=[1,0])";
    //     }
    // };

    // template <typename T>
    // class Expand : public Op
    // {
    // public:
    //     Expand()
    //     {
    //         this->init("expand", "any", {}, {}, false, {}, {});
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto input = mem.gettensor<T>(this->args[0]).get();
    //         auto output = mem.gettensor<T>(this->returns[0]).get();
    //         tensorfunc::expand(*input, *output);
    //     }
    //     vector<int> sumaxis(const vector<int>  shape,const vector<int> target_shape)
    //     {
    //         vector<int> axis;

    //         // 检查当前形状(this->shape)与目标形状的差异
    //         int current_dim =  shape.size();
    //         int target_dim = target_shape.size();

    //         // 如果目标维度小于当前维度，需要在一些轴上求和来减少维度
    //         if (target_dim < current_dim)
    //         {
    //             // 检查每个当前维度，看是否需要在目标形状中保留
    //             for (int i = 0; i < current_dim; i++)
    //             {
    //                 bool keep_dim = false;

    //                 // 找出当前维度是否与目标形状中的任何维度对应
    //                 if (i < current_dim - target_dim)
    //                 {
    //                     // 如果当前维度索引小于两者维度差，肯定需要被求和
    //                     axis.push_back(i);
    //                 }
    //                 else
    //                 {
    //                     // 检查该维度是否与目标形状匹配
    //                     int target_idx = i - (current_dim - target_dim);
    //                     if (target_shape[target_idx] == 1 &&  shape[i] > 1)
    //                     {
    //                         // 如果目标形状在这个维度上是1，但当前形状不是1，需要求和
    //                         axis.push_back(i);
    //                     }
    //                 }
    //             }
    //         }
    //         else if (target_dim == current_dim)
    //         {
    //             // 维度数量相同，检查哪些维度需要被压缩为1
    //             for (int i = 0; i < current_dim; i++)
    //             {
    //                 if (target_shape[i] == 1 && shape[i] > 1)
    //                 {
    //                     axis.push_back(i);
    //                 }
    //             }
    //         }
    //         // 如果目标维度大于当前维度，可能需要扩展维度(通常通过其他操作如expand_dims)

    //         return axis;
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto input_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto output_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         vector<int> target_shape = this->getvector<int32_t>( 1);
    //         vector<int> axis = this->sumaxis(input_grad->shape.shape,target_shape);
    //         // sum,按指定维度求和
    //         tensorfunc::sum(*output_grad,  axis,*input_grad);
    //     }
    //     void funcdef() override
    //     {
    //         this->init("expand", "float32", {"T1", "4", "6", "12"}, {"T2"}, false, {}, {});
    //     }
    //     string math_formula() const override
    //     {
    //         return "T2 = expand(T1, axis=[4,6,12])";
    //     }
    // };
}
#endif // DEEPX_OP_CONCAT_HPP