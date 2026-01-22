#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace memcuda {

class RedisClient {
public:
    explicit RedisClient(const std::string& uri);
    ~RedisClient();

    bool HSet(const std::string& key, const std::string& field, const std::string& value);
    bool HGet(const std::string& key, const std::string& field, std::string& out) const;
    std::unordered_map<std::string, std::string> HGetAll(const std::string& key) const;

    long long LPush(const std::string& key, const std::string& value);
    bool BRPop(const std::string& key, int timeout_seconds, std::string& out) const;

    std::string Eval(const std::string& script,
                     const std::vector<std::string>& keys,
                     const std::vector<std::string>& args) const;

private:
    void* redis_impl_;
};

}
