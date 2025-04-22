#ifndef DEEPX_MEM_MEMBASE_HPP
#define DEEPX_MEM_MEMBASE_HPP

#include <any>
#include <unordered_map>
#include <vector>
#include <memory>
#include "iostream"

#include "deepx/tensor.hpp"
namespace deepx::mem
{
    using namespace std;

    class MemBase
    {
    protected:
        unordered_map<string, std::any> args;
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> mem;
        int tempidx = 0;

    public:
        // 基本操作接口
        virtual void clear()
        {
            args.clear();
            mem.clear();
        }

        // 通用的arg操作
        template <typename T>
        void addarg(const string &name, const T value)
        {
            if (args.find(name) != args.end())
            {
                cerr << "arg already exists: " << name << endl;
            }
            args[name] = value;
        }

        template <typename T>
        T getarg(const string &name) const
        {
            if (args.find(name) == args.end())
            {
                cerr << "arg not found: " << name << endl;
                return T();
            }
            return any_cast<T>(args.at(name));
        }

        template <typename T>
        void addvector(const string &name, const vector<T> &value)
        {
            if (args.find(name) != args.end())
            {
                cerr << "vector already exists: " << name << endl;
                return;
            }
            args[name] = value;
        }

        template <typename T>
        vector<T> getvector(const string &name) const
        {
            if (args.find(name) == args.end())
            {
                cerr << "vector not found: " << name << endl;
                return vector<T>();
            }
            auto v = any_cast<vector<T>>(args.at(name));
            return v;
        }

        // tensor

        template <typename T>
        void addtensor(const string &name, Tensor<T> &&tensor)
        {
            if (mem.find(name) != mem.end())
            {
                cerr << "tensor already exists: " << name << endl;
                return;
            }
            auto ptr = std::make_shared<Tensor<T>>(std::move(tensor));
            mem[name] = ptr;
        }

        template <typename T>
        void addtensor(const string &name, const Tensor<T> &tensor)
        {
            if (mem.find(name) != mem.end())
            {
                cerr << "tensor already exists: " << name << endl;
                return;
            }
            auto ptr = std::make_shared<Tensor<T>>(tensor);
            mem[name] = ptr;
        }

        template <typename T>
        void addtensor(const string &name, shared_ptr<Tensor<T>> tensor)
        {
            if (mem.find(name) != mem.end())
            {
                cerr << "tensor already exists: " << name << endl;
                return;
            }
            mem[name] = tensor;
        }
 

        bool existstensor(const string &name) const
        {
            return mem.find(name) != mem.end();
        }
        bool existarg(const string &name) const
        {
            return args.find(name) != args.end();
        }

        template <typename T>
        shared_ptr<Tensor<T>> gettensor(const string &name) const
        {
            if (mem.find(name) == mem.end())
            {
                throw std::runtime_error("tensor not found: " + name);
            }
            auto ptr = mem.at(name);
            return std::static_pointer_cast<Tensor<T>>(ptr);
        }

        virtual shared_ptr<Tensor<void>> gettensor(const string &name) const = 0;

        // 获取多个张量
        template <typename T>
        vector<Tensor<T> *> gettensors(const vector<string> &names) const
        {
            std::vector<Tensor<T> *> tensors;

            for (const auto &name : names)
            {
                if (mem.find(name) == mem.end())
                {
                    cerr << "tensor not found: " << name << endl;
                    continue;
                }
                auto ptr = mem.at(name);
                tensors.push_back(std::static_pointer_cast<Tensor<T>>(ptr).get());
            }

            return tensors;
        }
 

        void delete_tensor(const string &name)
        {
            mem.erase(name);
        }

        void delete_arg(const string &name)
        {
            args.erase(name);
        }

        void rename_tensor(const string &old_name, const string &new_name)
        {
            if (mem.find(old_name) == mem.end())
            {
                throw std::runtime_error("tensor not found: " + old_name);
            }
            mem[new_name] = mem[old_name];  
            mem.erase(old_name);
        }
    };
}
#endif // DEEPX_MEM_MEMBASE_HPP