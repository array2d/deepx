#ifndef DEEPX_MEM_MEMOMPSIMD_HPP
#define DEEPX_MEM_MEMOMPSIMD_HPP

#include <any>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <memory>
#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
namespace deepx::mem
{
    using namespace std;
    class Mem : public MemBase
    {
    private:
    public:
        Mem() = default;
        ~Mem() = default;
        Mem(const Mem &other)
        {
            args = other.args;
            this->mem = other.mem;
        }
        Mem(Mem &&other) noexcept
        {
            args = std::move(other.args);
            this->mem = std::move(other.mem);
        }
        Mem &operator=(const Mem &other)
        {
            args = other.args;
            this->mem = other.mem;
            return *this;
        }
        Mem &operator=(Mem &&other) noexcept
        {
            args = std::move(other.args);
            this->mem = std::move(other.mem);
            return *this;
        }

        // TODO
        shared_ptr<Tensor<void>> gettensor(const string &name) const
        {
            if (mem.find(name) == mem.end())
            {
                return nullptr;
            }
            auto ptr = mem.at(name);
            auto result = make_shared<Tensor<void>>();
            result->shape = ptr->shape;
            result->deleter = nullptr;
            result->copyer = nullptr;
            result->newer = nullptr;

            switch (ptr->shape.dtype)
            {
            case Precision::Float64:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<double>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::Float32:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<float>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::Int64:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<int64_t>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::Int32:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<int32_t>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::Int16:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<int16_t>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::Int8:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<int8_t>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }

            default:
                throw std::runtime_error("Unsupported dtype: " + precision_str(ptr->shape.dtype));
            }

            return result;
        }
    };
}
#endif // DEEPX_MEM_MEM_HPP