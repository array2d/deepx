#pragma once

#include "../registry/registry.h"
#include "../shmem/shm_tensor.h"
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace deepx::heapmem {

struct LifecycleCommand {
    std::string op;      // "create", "get", "delete"
    std::string name;    // tensor name
    std::string dtype;
    std::string shape;
    int64_t     device = 0;
    int64_t     byte_size = 0;
    int64_t     pid = 0;
    int64_t     element_count = 0;
};

class LifecycleManager {
public:
    LifecycleManager(Registry *registry);

    // 处理一条指令，返回 shm_name（create/get 时有效）
    std::string handle(const LifecycleCommand &cmd, std::string &error);

    // 获取已打开的 shm tensor 的地址（供本地访问）
    void *get_addr(const std::string &shm_name) const;

    // 关闭所有已打开的 tensor
    void shutdown();

private:
    std::string generate_shm_name() const;

    Registry *registry_;
    std::unordered_map<std::string, deepx::shmem::ShmTensor> open_tensors_;
    mutable std::mutex mutex_;
    mutable std::atomic<uint64_t> counter_{0};
};

} // namespace deepx::heapmem
