#ifndef DEEPX_TF_IO_HPP
#define DEEPX_TF_IO_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/io.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tf
{

    template <typename Author>
    class Print : public TF
    {
    public:
        Print(vector<Param> args, vector<Param> returns)
        {
            this->name = "print";
            this->author = Author::name();
            this->tftype = "io";
            this->args = args;
            this->returns = returns;
        }
 
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            if (mem->existstensor(name))
            {
                auto t = mem->gettensor(name);
                if (this->args.size() == 1)
                {
                    tensorfunc::print<Author, void>(*t);
                }
                else
                {
                    tensorfunc::print<Author, void>(*t, this->args[1].textvalue);
                }
            }
            else
            {
                std::cerr << "print " << name << " not found" << std::endl;
                error = "print " + name + " not found";
                return 1;
            }
            return 0;
        }

        string math_formula() const override
        {
            return "print(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Print<Author>>(*this);
        }
    };

     //save
    class Save : public TF
    {
    public:
        Save(vector<Param> args, vector<Param> returns) 
        {
            this->name = "save";
            this->tftype = "io";
            this->args = args;
            this->returns = returns;
        }   
         string math_formula() const override
        {
            return "save(T1,path)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Save>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            string path = this->args[1].textvalue;
            if (mem->existstensor(name))
            {
                auto t = mem->gettensor(name);
                tensorfunc::save<void>(*t, path);
            }
            else
            {
                std::cerr << "save " << name << " not found" << std::endl;
                error = "save " + name + " not found";
                return 1;
            }
            return 0;
        }
    };

    //load
    class Load : public TF
    {
    public:
        Load(vector<Param> args, vector<Param> returns)
        {   
            this->name = "load";
            this->tftype = "io";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "load(path)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Load>(*this);
        }
        int run(shared_ptr<MemBase> mem, string &error) override    
        {
            string path = this->args[0].textvalue;
            
            pair<std::string,Shape> shape_name=tensorfunc::loadShape(path);
            std::string tensor_name=shape_name.first;
            Shape shape=shape_name.second;

            if(mem->existstensor(tensor_name))
            {
                cout<<"warning: "<<tensor_name<<" already exists,deepx will delete it,create new one"<<endl;
                mem->delete_tensor(tensor_name);
            }
            switch (shape.dtype)
            {
            case Precision::Float64:{
                pair<std::string,shared_ptr<Tensor<double>>> t = tensorfunc::load<double>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Float32:{
                pair<std::string,shared_ptr<Tensor<float>>> t = tensorfunc::load<float>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Float16:{
                pair<std::string,shared_ptr<Tensor<half>>> t = tensorfunc::load<half>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::BFloat16:{
                pair<std::string,shared_ptr<Tensor<nv_bfloat16>>> t = tensorfunc::load<nv_bfloat16>(path);
                mem->addtensor(tensor_name, t.second);
                break;  
            }
            case Precision::Int64:{
                pair<std::string,shared_ptr<Tensor<int64_t>>> t = tensorfunc::load<int64_t>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Int32:{
                pair<std::string,shared_ptr<Tensor<int32_t>>> t = tensorfunc::load<int32_t>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Int16:{
                pair<std::string,shared_ptr<Tensor<int16_t>>> t = tensorfunc::load<int16_t>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Int8:{
                pair<std::string,shared_ptr<Tensor<int8_t>>> t = tensorfunc::load<int8_t>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            case Precision::Bool:{
                pair<std::string,shared_ptr<Tensor<bool>>> t = tensorfunc::load<bool>(path);
                mem->addtensor(tensor_name, t.second);
                break;
            }
            default:
                break;
            }
            return 0;
        }
    };
}
#endif
