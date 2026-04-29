#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "redis_client.h"

namespace memcuda {

struct TensorMeta {
    std::string dtype;
    std::string shape;
    std::string node;
    std::string ipc_handle;
    long long device = 0;
    long long bytes = 0;
    long long refcount = 0;
    long long owner_pid = 0;
    long long ctime = 0;
    std::string state;
};

class Registry {
public:
    Registry(RedisClient* client, const std::string& lua_dir);

    std::string CreateOrGet(const std::string& name,
                            const std::string& dtype,
                            const std::string& shape,
                            long long device,
                            long long bytes,
                            const std::string& node,
                            long long pid,
                            long long ctime,
                            const std::string& ipc_handle);

    long long RefInc(const std::string& name);
    long long RefDec(const std::string& name);
    long long GcSweep(const std::string& node);

    bool GetMeta(const std::string& name, TensorMeta& out) const;
    RedisClient* Client() const;

private:
    std::string Script(const std::string& file) const;
    std::string Key(const std::string& name) const;

    RedisClient* client_;
    std::string lua_dir_;
};

}
