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
            this->args = args;
            this->returns = returns;
        }
        Print(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "print")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
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
}
#endif
