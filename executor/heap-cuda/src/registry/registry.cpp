#include "registry.h"

#include <fstream>
#include <sstream>

namespace memcuda {

Registry::Registry(RedisClient* client, const std::string& lua_dir)
    : client_(client), lua_dir_(lua_dir) {}

std::string Registry::Key(const std::string& name) const {
    return "tensor:" + name;
}

std::string Registry::Script(const std::string& file) const {
    std::ifstream in(lua_dir_ + "/" + file, std::ios::in | std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string Registry::CreateOrGet(const std::string& name,
                                  const std::string& dtype,
                                  const std::string& shape,
                                  long long device,
                                  long long bytes,
                                  const std::string& node,
                                  long long pid,
                                  long long ctime,
                                  const std::string& ipc_handle) {
    std::vector<std::string> keys{Key(name)};
    std::vector<std::string> args{
        dtype,
        shape,
        std::to_string(device),
        std::to_string(bytes),
        node,
        std::to_string(pid),
        std::to_string(ctime),
        ipc_handle
    };
    return client_->Eval(Script("create_or_get.lua"), keys, args);
}

long long Registry::RefInc(const std::string& name) {
    std::vector<std::string> keys{Key(name)};
    std::vector<std::string> args;
    auto res = client_->Eval(Script("ref_inc.lua"), keys, args);
    return std::stoll(res);
}

long long Registry::RefDec(const std::string& name) {
    std::vector<std::string> keys{Key(name)};
    std::vector<std::string> args;
    auto res = client_->Eval(Script("ref_dec.lua"), keys, args);
    return std::stoll(res);
}

long long Registry::GcSweep(const std::string& node) {
    std::vector<std::string> keys;
    std::vector<std::string> args{node};
    auto res = client_->Eval(Script("gc_sweep.lua"), keys, args);
    return std::stoll(res);
}

bool Registry::GetMeta(const std::string& name, TensorMeta& out) const {
    auto map = client_->HGetAll(Key(name));
    if (map.empty()) {
        return false;
    }
    out.dtype = map["dtype"];
    out.shape = map["shape"];
    out.node = map["node"];
    out.ipc_handle = map["ipc_handle"];
    out.state = map["state"];
    if (map.count("device")) out.device = std::stoll(map["device"]);
    if (map.count("bytes")) out.bytes = std::stoll(map["bytes"]);
    if (map.count("refcount")) out.refcount = std::stoll(map["refcount"]);
    if (map.count("owner_pid")) out.owner_pid = std::stoll(map["owner_pid"]);
    if (map.count("ctime")) out.ctime = std::stoll(map["ctime"]);
    return true;
}

RedisClient* Registry::Client() const {
    return client_;
}

}
