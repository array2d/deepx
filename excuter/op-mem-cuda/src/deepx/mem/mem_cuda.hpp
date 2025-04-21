#ifndef DEEPX_MEM_MEMCUDA_HPP
#define DEEPX_MEM_MEMCUDA_HPP

#include <any>
#include <unordered_map>
#include <vector>
#include <memory>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
namespace deepx::mem
{
    using namespace std;

    class Mem : public MemBase
    {
    public:
        Mem() = default;
        ~Mem() = default;
        Mem(const Mem &other)
        {
            args = other.args;
            mem = other.mem;
        }
        Mem(Mem &&other) noexcept
        {
            args = std::move(other.args);
            mem = std::move(other.mem);
        }
        Mem &operator=(const Mem &other)
        {
            args = other.args;
            mem = other.mem;
            return *this;
        }
        Mem &operator=(Mem &&other) noexcept
        {
            args = std::move(other.args);
            mem = std::move(other.mem);
            return *this;
        }

        shared_ptr<Tensor<void>> gettensor(const string &name) const override
        {
            if (mem.find(name) == mem.end())
            {
                throw std::runtime_error("tensor not found: " + name);
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
            case Precision::Float16:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<__half>>(ptr);
                result->data = ptr_tensor->data;
                break;
            }
            case Precision::BFloat16:
            {
                auto ptr_tensor = std::static_pointer_cast<Tensor<__nv_bfloat16>>(ptr);
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