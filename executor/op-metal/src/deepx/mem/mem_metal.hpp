#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>

#include "tensor.hpp"
#include "mem/mem.hpp"
#include "deepx/shmem/shm_tensor.h"

namespace deepx::mem {

// Metal 侧 Mem 实现 — tensor 数据来自 POSIX shm (heap-metal 分配)
class MemMetal : public MemBase {
public:
    MemMetal() = default;

    // 注册一个 tensor 的 shm 映射
    // shm_name: POSIX shm 名称
    // byte_size: tensor 数据字节数
    // addr: mmap 后的虚拟地址
    void register_tensor(const std::string &name,
                         const std::string &shm_name,
                         size_t byte_size,
                         void *addr,
                         const Shape &shape) {
        auto info = std::make_shared<TensorInfo>();
        info->shm_name  = shm_name;
        info->byte_size = byte_size;
        info->addr      = addr;
        info->shape     = shape;
        tensors_[name]  = info;
    }

    // 导入 heap-metal 已创建的 tensor
    bool import(const std::string &name, const std::string &shm_name,
                size_t byte_size, const Shape &shape) {
        deepx::shmem::ShmTensor st;
        if (!deepx::shmem::shm_tensor_open(shm_name, byte_size, st)) {
            return false;
        }
        register_tensor(name, shm_name, byte_size, st.addr, shape);
        return true;
    }

    std::shared_ptr<Tensor<void>> gettensor(const std::string &name) const override {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("tensor not found: " + name);
        }
        auto &info = it->second;
        auto result = std::make_shared<Tensor<void>>();
        result->shape   = info->shape;
        result->data    = info->addr;
        result->deleter = nullptr; // shm 生命周期由 heap 管理
        result->copyer  = nullptr;
        result->newer   = nullptr;
        return result;
    }

    // 本地创建（单进程调试用，不走 shm）
    template <typename T>
    void local_new(const std::string &name, const std::vector<int> &dims) {
        Shape shape(dims);
        shape.dtype = precision<T>();
        size_t bytes = shape.size * sizeof(T);
        T *data = new T[shape.size];

        auto info = std::make_shared<TensorInfo>();
        info->shm_name  = ""; // 本地分配
        info->byte_size = bytes;
        info->addr      = data;
        info->shape     = shape;
        info->local     = true;
        tensors_[name]  = info;
    }

    ~MemMetal() {
        for (auto &kv : tensors_) {
            auto &info = kv.second;
            if (!info->shm_name.empty()) {
                deepx::shmem::ShmTensor st;
                st.shm_name  = info->shm_name;
                st.addr      = info->addr;
                st.byte_size = info->byte_size;
                deepx::shmem::shm_tensor_close(st);
            } else if (info->local && info->addr) {
                // 本地 new[] 的释放
                operator delete(info->addr);
            }
        }
    }

private:
    struct TensorInfo {
        std::string shm_name;
        size_t      byte_size = 0;
        void       *addr = nullptr;
        Shape       shape;
        bool        local = false; // true 表示本地 new[]，需自行释放
    };
    std::unordered_map<std::string, std::shared_ptr<TensorInfo>> tensors_;
};

} // namespace deepx::mem
