#ifndef DEEPX_TF_IO_HPP
#define DEEPX_TF_IO_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/io.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    using namespace std;

    template <typename Author>
    class Print : public TF
    {
    public:
        Print(vector<Param> args, vector<Param> returns)
        {
            this->name = "print";
            this->metadata.author = Author::name();
            this->tftype = "io";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override { return "print(T1)"; }
        shared_ptr<TF> clone() const override { return make_shared<Print<Author>>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            if (!mem->existstensor(name)) {
                error = "print " + name + " not found"; return 1;
            }
            string format = (this->args.size() > 1) ? this->args[1].textvalue : "";
            Precision dtype = mem->gettensor(name)->shape.dtype;
            switch (dtype) {
            case Precision::Float64:{ auto t=mem->gettensor<double>(name); print<Author,double>(*t,format); break; }
            case Precision::Float32:{ auto t=mem->gettensor<float>(name); print<Author>(*t,format); break; }
            case Precision::Int64:  { auto t=mem->gettensor<int64_t>(name); print<Author>(*t,format); break; }
            case Precision::Int32:  { auto t=mem->gettensor<int32_t>(name); print<Author>(*t,format); break; }
            case Precision::Int16:  { auto t=mem->gettensor<int16_t>(name); print<Author>(*t,format); break; }
            case Precision::Int8:   { auto t=mem->gettensor<int8_t>(name); print<Author>(*t,format); break; }
            case Precision::Bool:   { auto t=mem->gettensor<bool>(name); print<Author,bool>(*t,format); break; }
            default: break;
            }
            return 0;
        }
    };

    // save
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
        string math_formula() const override { return "save(T1,path)"; }
        shared_ptr<TF> clone() const override { return make_shared<Save>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string name = this->args[0].textvalue;
            string path = this->args[1].textvalue;
            if (!mem->existstensor(name)) {
                error = "save " + name + " not found"; return 1;
            }
            Precision dtype = mem->gettensor(name)->shape.dtype;
            mem->gettensor(name)->shape.saveShape(path);
            path += ".data";
            switch (dtype) {
            case Precision::Float64:{ auto t=mem->gettensor<double>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Float32:{ auto t=mem->gettensor<float>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Int64:  { auto t=mem->gettensor<int64_t>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Int32:  { auto t=mem->gettensor<int32_t>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Int16:  { auto t=mem->gettensor<int16_t>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Int8:   { auto t=mem->gettensor<int8_t>(name); t->saver(t->data,t->shape.size,path); break; }
            case Precision::Bool:   { auto t=mem->gettensor<bool>(name); t->saver(t->data,t->shape.size,path); break; }
            default: break;
            }
            return 0;
        }
    };

    // load
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
        string math_formula() const override { return "load(path)"; }
        shared_ptr<TF> clone() const override { return make_shared<Load>(*this); }
        int run(shared_ptr<MemBase> mem, string &error) override
        {
            string path = this->args[0].textvalue;
            pair<string,Shape> shape_name = Shape::loadShape(path);
            string tensor_name = shape_name.first;
            Shape shape = shape_name.second;
            if (mem->existstensor(tensor_name)) {
                cout << "warning: " << tensor_name << " already exists, replacing" << endl;
                mem->delete_tensor(tensor_name);
            }
            switch (shape.dtype) {
            case Precision::Float64:{ auto t=tensorfunc::load<double>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Float32:{ auto t=tensorfunc::load<float>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Int64:  { auto t=tensorfunc::load<int64_t>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Int32:  { auto t=tensorfunc::load<int32_t>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Int16:  { auto t=tensorfunc::load<int16_t>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Int8:   { auto t=tensorfunc::load<int8_t>(path); mem->addtensor(tensor_name, t.second); break; }
            case Precision::Bool:   { auto t=tensorfunc::load<bool>(path); mem->addtensor(tensor_name, t.second); break; }
            default: break;
            }
            return 0;
        }
    };

} // namespace deepx::tf

#endif // DEEPX_TF_IO_HPP
