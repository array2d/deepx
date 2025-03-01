#ifndef DEEPX_OP_NEW_HPP
#define DEEPX_OP_NEW_HPP

#include "deepx/op/op.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "stdutil/num.hpp"

namespace deepx::op{
    template<typename T>
    class NewTensor : public OpT<T>{
        public:
        NewTensor(){
            this->init("newtensor",dtype<T>::name(), {}, {}, false, {}, {});
        }
        NewTensor(string args){
            this->init("newtensor",dtype<T>::name(), args, {}, false, {}, {});
        }
        NewTensor(initializer_list<string> args){
            this->init("newtensor",dtype<T>::name(), args, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override{
            string name= this->returns[0];
            if (this->args.size()==1&& !is_positive_integer(this->args[0])){
                vector<int> shape=mem.getvector<int32_t>(this->args[0]);
                Tensor<T> t=tensorfunc::New<T>(shape);
                mem.addtensor(name,t);
            }else if (this->args.size()>1){
                vector<int> shape;  
                for (int i = 0; i < this->args.size(); i++) {
                    shape.push_back(atoi(this->args[i].c_str()));
                }
                Tensor<T> t=tensorfunc::New<T>(shape);
                mem.addtensor(name,t);
            }
        }   
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("New op does not support backward");
        }
    };



    template<typename T>
    class DelTensor : public OpT<T>{
        public:
        DelTensor(){
            this->init("deltensor","any", {}, {}, false, {}, {});
        }
        DelTensor(string args){
            this->init("deltensor","any", args, {}, false, {}, {});
        }
        DelTensor(initializer_list<string> args){
            this->init("deltensor","any", args, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override{
            string name= this->args[0];
            mem.delete_tensor<T>(name);
        }
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Del op does not support backward");
        }
    };
}
#endif
