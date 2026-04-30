#pragma once

#include "deepx/registry.h"
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <sstream>

namespace deepx::heap {

// 基于文件的简单 Registry 实现（验证用）
// 生产环境替换为 RedisRegistry
class FileRegistry : public Registry {
public:
    explicit FileRegistry(const std::string &path) : path_(path) {
        load();
    }

    ~FileRegistry() override { save(); }

    std::string create_or_get(const std::string &name,
                              const std::string &dtype,
                              const std::string &shape,
                              int64_t device,
                              int64_t byte_size,
                              int64_t pid,
                              const std::string &shm_name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = store_.find(name);
        if (it != store_.end()) {
            it->second.refcount++;
            save();
            return it->second.shm_name;
        }
        TensorMeta meta;
        meta.name      = name;
        meta.shm_name  = shm_name;
        meta.dtype     = dtype;
        meta.shape     = shape;
        meta.device    = device;
        meta.byte_size = byte_size;
        meta.owner_pid = pid;
        meta.refcount  = 1;
        meta.ctime     = time(nullptr);
        meta.state     = "ready";
        store_[name] = meta;
        save();
        return shm_name;
    }

    int64_t ref_inc(const std::string &name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = store_.find(name);
        if (it == store_.end()) return -1;
        it->second.refcount++;
        save();
        return it->second.refcount;
    }

    int64_t ref_dec(const std::string &name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = store_.find(name);
        if (it == store_.end()) return -1;
        it->second.refcount--;
        if (it->second.refcount <= 0) {
            it->second.state = "deleted";
        }
        save();
        return it->second.refcount;
    }

    bool get_meta(const std::string &name, TensorMeta &out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = store_.find(name);
        if (it == store_.end()) return false;
        out = it->second;
        return true;
    }

private:
    void load() {
        std::ifstream f(path_);
        if (!f) return;
        std::string line;
        while (std::getline(f, line)) {
            auto pos = line.find(' ');
            if (pos == std::string::npos) continue;
            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);
            if (key == "tensor") {
                TensorMeta meta;
                // simple format: tensor name shm_name dtype shape device bytes refcount pid ctime state
                std::istringstream iss(val);
                iss >> meta.name >> meta.shm_name >> meta.dtype >> meta.shape
                    >> meta.device >> meta.byte_size >> meta.refcount
                    >> meta.owner_pid >> meta.ctime >> meta.state;
                if (!meta.name.empty()) {
                    store_[meta.name] = meta;
                }
            }
        }
    }

    void save() {
        std::ofstream f(path_, std::ios::trunc);
        if (!f) return;
        for (auto &kv : store_) {
            auto &m = kv.second;
            f << "tensor " << m.name << " " << m.shm_name << " " << m.dtype << " "
              << m.shape << " " << m.device << " " << m.byte_size << " "
              << m.refcount << " " << m.owner_pid << " " << m.ctime << " "
              << m.state << "\n";
        }
    }

    std::string path_;
    std::unordered_map<std::string, TensorMeta> store_;
    std::mutex mutex_;
};

} // namespace deepx::heap
