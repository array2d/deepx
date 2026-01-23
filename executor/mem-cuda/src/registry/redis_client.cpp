#include "redis_client.h"

#include <sw/redis++/redis++.h>

namespace memcuda {

RedisClient::RedisClient(const std::string& uri) {
    auto* r = new sw::redis::Redis(uri);
    redis_impl_ = static_cast<void*>(r);
}

RedisClient::~RedisClient() {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    delete r;
}

bool RedisClient::HSet(const std::string& key, const std::string& field, const std::string& value) {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    return r->hset(key, field, value);
}

bool RedisClient::HGet(const std::string& key, const std::string& field, std::string& out) const {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    auto val = r->hget(key, field);
    if (!val) {
        return false;
    }
    out = *val;
    return true;
}

std::unordered_map<std::string, std::string> RedisClient::HGetAll(const std::string& key) const {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    std::unordered_map<std::string, std::string> res;
    r->hgetall(key, std::inserter(res, res.begin()));
    return res;
}

long long RedisClient::LPush(const std::string& key, const std::string& value) {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    return r->lpush(key, value);
}

bool RedisClient::BRPop(const std::string& key, int timeout_seconds, std::string& out) const {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    auto item = r->brpop(key, std::chrono::seconds(timeout_seconds));
    if (!item) {
        return false;
    }
    out = item->second;
    return true;
}

std::string RedisClient::Eval(const std::string& script,
                              const std::vector<std::string>& keys,
                              const std::vector<std::string>& args) const {
    auto* r = static_cast<sw::redis::Redis*>(redis_impl_);
    return r->eval<std::string>(script, keys.begin(), keys.end(), args.begin(), args.end());
}

}
