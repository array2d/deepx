#pragma once

#include <atomic>
#include <string>

#include "../registry/registry.h"

namespace memcuda {

struct LifecycleCommand {
    std::string op;
    std::string name;
    std::string dtype;
    std::string shape;
    std::string node;
    long long device = 0;
    long long pid = 0;
};

class LifecycleWorker {
public:
    LifecycleWorker(Registry* registry, const std::string& queue_key);

    void RunOnce(int timeout_seconds);
    void RunLoop(int timeout_seconds);
    void Stop();

private:
    bool Parse(const std::string& json, LifecycleCommand& out) const;
    void Handle(const LifecycleCommand& cmd);

    Registry* registry_;
    std::string queue_key_;
    std::atomic<bool> running_;
};

}
