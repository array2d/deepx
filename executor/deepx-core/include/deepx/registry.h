#pragma once

#include <string>
#include <cstdint>

namespace deepx::heap {

// 一个 tensor 的 Redis 元数据
struct TensorMeta {
    std::string name;        // tensor name
    std::string shm_name;    // POSIX shm name (e.g. "/deepx_t_abc123")
    std::string dtype;       // "f32", "i32", etc.
    std::string shape;       // "[2,3,4]"
    int64_t     device = 0;
    int64_t     byte_size = 0;
    int64_t     refcount = 0;
    int64_t     owner_pid = 0;
    int64_t     ctime = 0;
    std::string state;       // "ready", "deleted"
};

// Registry 接口 — 抽象 Redis 后端。
// 当前实现可以是 Redis，后续可替换为 etcd/文件等。
class Registry {
public:
    virtual ~Registry() = default;

    // 创建或获取一个 tensor。返回 shm_name。
    // 如果 tensor 已存在，增加引用计数。
    virtual std::string create_or_get(const std::string &name,
                                      const std::string &dtype,
                                      const std::string &shape,
                                      int64_t device,
                                      int64_t byte_size,
                                      int64_t pid,
                                      const std::string &shm_name) = 0;

    // 引用计数 +1
    virtual int64_t ref_inc(const std::string &name) = 0;

    // 引用计数 -1；若为 0 则标记可回收
    virtual int64_t ref_dec(const std::string &name) = 0;

    // 获取 tensor 元数据
    virtual bool get_meta(const std::string &name, TensorMeta &out) = 0;
};

} // namespace deepx::heap
