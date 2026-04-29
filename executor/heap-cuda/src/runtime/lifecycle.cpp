#include "lifecycle.h"

#include <cctype>

namespace memcuda {

static std::string ExtractString(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        pos++;
        auto end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }
    auto end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && !std::isspace(static_cast<unsigned char>(json[end]))) end++;
    return json.substr(pos, end - pos);
}

static long long ExtractInt(const std::string& json, const std::string& key) {
    auto s = ExtractString(json, key);
    if (s.empty()) return 0;
    return std::stoll(s);
}

LifecycleWorker::LifecycleWorker(Registry* registry, const std::string& queue_key)
    : registry_(registry), queue_key_(queue_key), running_(false) {}

bool LifecycleWorker::Parse(const std::string& json, LifecycleCommand& out) const {
    out.op = ExtractString(json, "op");
    out.name = ExtractString(json, "name");
    out.dtype = ExtractString(json, "dtype");
    out.shape = ExtractString(json, "shape");
    out.node = ExtractString(json, "node");
    out.device = ExtractInt(json, "device");
    out.pid = ExtractInt(json, "pid");
    return !out.op.empty() && !out.name.empty();
}

void LifecycleWorker::Handle(const LifecycleCommand& cmd) {
    if (cmd.op == "create") {
        registry_->CreateOrGet(cmd.name, cmd.dtype, cmd.shape, cmd.device, 0, cmd.node, cmd.pid, 0, "");
        return;
    }
    if (cmd.op == "get") {
        registry_->RefInc(cmd.name);
        return;
    }
    if (cmd.op == "delete") {
        registry_->RefDec(cmd.name);
        return;
    }
}

void LifecycleWorker::RunOnce(int timeout_seconds) {
    if (!registry_) return;
    auto* client = registry_->Client();
    if (!client) return;
    std::string msg;
    if (!client->BRPop(queue_key_, timeout_seconds, msg)) {
        return;
    }
    LifecycleCommand cmd;
    if (!Parse(msg, cmd)) {
        return;
    }
    Handle(cmd);
}

void LifecycleWorker::RunLoop(int timeout_seconds) {
    running_.store(true);
    while (running_.load()) {
        RunOnce(timeout_seconds);
    }
}

void LifecycleWorker::Stop() {
    running_.store(false);
}

}
