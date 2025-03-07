#ifndef DEEPX_OP_INIT_HPP
#define DEEPX_OP_INIT_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "stdutil/num.hpp"
namespace deepx::op{
    template<typename T>
    class Uniform : public Op{
        public:
        Uniform(){
            this->init("uniform",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        Uniform(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("uniform",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Uniform(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("uniform",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
            T low =  this->getarg<T>(0,mem);
            T high = this->getarg<T>(1,mem);
            uint32_t seed = 0;
            if (this->args.size() == 3){
                seed = this->getarg<uint32_t>(2,mem);
            }
            tensorfunc::uniform(*output,low,high,seed);
        } 
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Uniform op does not support backward");
        }
        void setexample() override {
            this->init("uniform", "float32", {"-1.0", "1.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "uniform(-1.0, 1.0,T1)";  // 均匀分布初始化
        }
    };

    template<typename T>
    class Constant : public Op{
        public:
        Constant(){
            this->init("constant",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        Constant(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("constant",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Constant(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("constant",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
             
            T value = this->getarg<T>(0,mem);
            tensorfunc::constant(*output,value);
        }
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Constant op does not support backward");
        }
        void setexample() override {
            this->init("constant", "float32", {"0.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "T1 = full(shape, 0.0)";  // 常量初始化
        }
    };

    template<typename T>
    class Arange : public Op{
        public:
        Arange(){
            this->init("arange",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        Arange(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("arange",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Arange(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("arange",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
            T start =  this->getarg<T>(0,mem);
            T step =  this->getarg<T>(1,mem);
            tensorfunc::arange(*output,start,step);
        }
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Arange op does not support backward");
        }
        void setexample() override {
            this->init("arange", "float32", {"0.0","1.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "arange(start=0.0, step=1.0,T1)";  // 等差数列
        }
    };
}

#endif
