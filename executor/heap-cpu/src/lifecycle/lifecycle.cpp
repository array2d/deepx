#include "lifecycle.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstdio>

namespace deepx::heap {

int64_t parse_shape_size(const std::string &shape_str) {
    // "[2,3,4]" → 2*3*4 = 24
    int64_t total = 1;
    int64_t cur = 0;
    bool in_num = false;
    for (char c : shape_str) {
        if (c >= '0' && c <= '9') {
            cur = cur * 10 + (c - '0');
            in_num = true;
        } else {
            if (in_num) {
                total *= cur;
                cur = 0;
                in_num = false;
            }
        }
    }
    if (in_num) total *= cur;
    return total;
}

LifecycleManager::LifecycleManager(Registry *registry)
    : registry_(registry) {}

std::string LifecycleManager::generate_shm_name() const {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t id = counter_.fetch_add(1);
    std::ostringstream oss;
    oss << "/deepx_t_" << std::hex << now << "_" << id;
    return oss.str();
}

std::string LifecycleManager::handle(const LifecycleCommand &cmd, std::string &error) {
    if (cmd.op == "newtensor") {
        // 计算 byte_size
        int64_t element_count = cmd.element_count > 0
            ? cmd.element_count
            : parse_shape_size(cmd.shape);

        // 确定每个元素的字节数
        int elem_bytes = 4; // 默认 f32
        if (cmd.dtype == "f64") elem_bytes = 8;
        else if (cmd.dtype == "f32") elem_bytes = 4;
        else if (cmd.dtype == "f16" || cmd.dtype == "bf16") elem_bytes = 2;
        else if (cmd.dtype == "i64") elem_bytes = 8;
        else if (cmd.dtype == "i32") elem_bytes = 4;
        else if (cmd.dtype == "i16") elem_bytes = 2;
        else if (cmd.dtype == "i8" || cmd.dtype == "bool") elem_bytes = 1;

        int64_t total_bytes = element_count * elem_bytes;

        // 检查是否已存在
        TensorMeta existing;
        if (registry_->get_meta(cmd.name, existing)) {
            // Tensor 已存在 → 打开已有 shm，ref_inc
            registry_->ref_inc(cmd.name);
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = open_tensors_.find(existing.shm_name);
            if (it == open_tensors_.end()) {
                deepx::shmem::ShmTensor st;
                if (!deepx::shmem::shm_tensor_open(existing.shm_name, existing.byte_size, st)) {
                    error = "failed to open existing shm: " + existing.shm_name;
                    return "";
                }
                open_tensors_[existing.shm_name] = st;
            }
            return existing.shm_name;
        }

        // 创建新的 shm tensor
        std::string shm_name = generate_shm_name();
        deepx::shmem::ShmTensor st;
        if (!deepx::shmem::shm_tensor_create(shm_name, total_bytes, st)) {
            error = "shm_tensor_create failed for " + shm_name;
            return "";
        }

        // 注册到 registry
        registry_->create_or_get(cmd.name, cmd.dtype, cmd.shape,
                                  cmd.device, total_bytes, cmd.pid, shm_name);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_tensors_[shm_name] = st;
        }

        printf("[heap] created tensor '%s' → shm=%s bytes=%lld\n",
               cmd.name.c_str(), shm_name.c_str(), total_bytes);
        return shm_name;
    }
    else if (cmd.op == "gettensor") {
        TensorMeta meta;
        if (!registry_->get_meta(cmd.name, meta)) {
            error = "tensor not found: " + cmd.name;
            return "";
        }
        registry_->ref_inc(cmd.name);

        // 确保已打开
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = open_tensors_.find(meta.shm_name);
        if (it == open_tensors_.end()) {
            deepx::shmem::ShmTensor st;
            if (!deepx::shmem::shm_tensor_open(meta.shm_name, meta.byte_size, st)) {
                error = "failed to open shm: " + meta.shm_name;
                return "";
            }
            open_tensors_[meta.shm_name] = st;
        }
        return meta.shm_name;
    }
    else if (cmd.op == "deltensor") {
        int64_t ref = registry_->ref_dec(cmd.name);
        printf("[heap] delete '%s' → refcount=%lld\n", cmd.name.c_str(), ref);

        if (ref <= 0) {
            TensorMeta meta;
            if (registry_->get_meta(cmd.name, meta)) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = open_tensors_.find(meta.shm_name);
                if (it != open_tensors_.end()) {
                    deepx::shmem::shm_tensor_close(it->second);
                    deepx::shmem::shm_tensor_unlink(it->second.shm_name);
                    open_tensors_.erase(it);
                }
            }
        }
        return "";
    }
    error = "unknown op: " + cmd.op;
    return "";
}

void *LifecycleManager::get_addr(const std::string &shm_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = open_tensors_.find(shm_name);
    if (it != open_tensors_.end()) {
        return it->second.addr;
    }
    return nullptr;
}

void LifecycleManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &kv : open_tensors_) {
        deepx::shmem::shm_tensor_close(kv.second);
    }
    open_tensors_.clear();
}

} // namespace deepx::heap
