#pragma once

#include "deepx/registry.h"
#include "deepx/shmem/shm_tensor.h"
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace deepx::heap {

// 从 "[10,20,30]" 解析 element count
int64_t parse_shape_size(const std::string &shape_str);

struct LifecycleCommand {
    std::string op;      // "newtensor", "gettensor", "deltensor"
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

} // namespace deepx::heap
