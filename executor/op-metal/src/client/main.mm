#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

#include "deepx/tensorfunc/elementwise_common.hpp"

using json = nlohmann::json;

static const char *OP_QUEUE     = "cmd:op-metal:0";
static const char *INSTANCE_KEY = "/sys/op-plat/op-metal:0";
static const int   BLOCK_TIMEOUT_SEC = 5;

// ═══════════════════════════════════════════════════════════
// Redis helpers
// ═══════════════════════════════════════════════════════════

static redisContext* connect_redis(const char *addr, int port) {
    struct timeval tv = {2, 0};
    redisContext *c = redisConnectWithTimeout(addr, port, tv);
    if (!c || c->err) {
        std::cerr << "[op-metal] Redis connect failed: " << (c ? c->errstr : "null") << "\n";
        if (c) redisFree(c);
        return nullptr;
    }
    return c;
}

static redisReply* redis_cmd(redisContext *c, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    redisReply *r = (redisReply *)redisvCommand(c, fmt, ap);
    va_end(ap);
    return r;
}

#define REDIS_FREE(r) do { if (r) freeReplyObject(r); } while(0)

static bool redis_set(redisContext *c, const std::string &key, const std::string &val) {
    redisReply *r = redis_cmd(c, "SET %s %s", key.c_str(), val.c_str());
    bool ok = r && r->type == REDIS_REPLY_STATUS;
    REDIS_FREE(r);
    return ok;
}

static void register_instance(redisContext *c) {
    json reg;
    reg["program"]    = "op-metal";
    reg["device"]     = "gpu0";
    reg["status"]     = "running";
    reg["load"]       = 0.0;
    reg["pid"]        = getpid();
    reg["started_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    redis_set(c, INSTANCE_KEY, reg.dump());
    std::cout << "[op-metal] registered at " << INSTANCE_KEY << "\n";

    // 注册支持的算子列表
    redisReply *r = redis_cmd(c, "DEL %s", "/op/op-metal/list");
    REDIS_FREE(r);
    redis_cmd(c, "RPUSH %s %s %s %s %s %s",
              "/op/op-metal/list", "add", "sub", "mul", "matmul", "relu");
    std::cout << "[op-metal] registered ops: add sub mul matmul relu\n";
}

static void notify_done(redisContext *c, const std::string &vtid,
                        const std::string &pc, const std::string &status,
                        const std::string &error_msg = "") {
    json done;
    done["pc"]     = pc;
    done["status"] = status;
    if (!error_msg.empty()) {
        done["error"] = {{"code", "OP_ERROR"}, {"message", error_msg}};
    }
    std::string key = "done:" + vtid;
    redisReply *r = redis_cmd(c, "LPUSH %s %s", key.c_str(), done.dump().c_str());
    if (!r || r->type == REDIS_REPLY_ERROR) {
        std::cerr << "[op-metal] notify_done LPUSH failed for " << vtid << ": " << (r ? r->str : "NULL") << "\n";
    }
    REDIS_FREE(r);
    std::cout << "[op-metal] done " << vtid << " pc=" << pc << " status=" << status << "\n";
}

// ═══════════════════════════════════════════════════════════
// shm helpers
// ═══════════════════════════════════════════════════════════

static size_t page_size() {
    static long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (size_t)ps : 16384;
}

static size_t page_align(size_t n) {
    size_t ps = page_size();
    return (n + ps - 1) & ~(ps - 1);
}

struct ShmMapping {
    std::string shm_name;
    void       *addr = nullptr;
    size_t      byte_size = 0;
    int         fd = -1;
};

static bool shm_open_readwrite(const std::string &name, size_t byte_size, ShmMapping &out) {
    out.shm_name  = name;
    out.byte_size = byte_size;

    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        std::cerr << "[op-metal] shm_open failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    size_t aligned = page_align(byte_size);
    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::cerr << "[op-metal] mmap failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    out.addr = addr;
    out.fd   = -1;
    return true;
}

static void shm_close(ShmMapping &m) {
    if (m.addr) {
        munmap(m.addr, page_align(m.byte_size));
        m.addr = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════
// Tensor metadata from Redis
// ═══════════════════════════════════════════════════════════

struct TensorMeta {
    std::string key;
    std::string dtype;
    std::vector<int64_t> shape;
    std::string shm_name;
    size_t      byte_size = 0;
    bool        valid = false;
};

static TensorMeta fetch_tensor_meta(redisContext *c, const std::string &key) {
    TensorMeta m;
    m.key = key;

    redisReply *r = redis_cmd(c, "GET %s", key.c_str());
    if (!r || r->type != REDIS_REPLY_STRING) {
        REDIS_FREE(r);
        return m;
    }

    try {
        json meta = json::parse(r->str);
        REDIS_FREE(r);

        if (meta.contains("dtype")) m.dtype = meta["dtype"].get<std::string>();
        if (meta.contains("shape") && meta["shape"].is_array()) {
            for (const auto &d : meta["shape"]) {
                m.shape.push_back(d.get<int64_t>());
            }
        }
        if (meta.contains("byte_size")) m.byte_size = meta["byte_size"].get<size_t>();
        if (meta.contains("address") && meta["address"].contains("shm_name")) {
            m.shm_name = meta["address"]["shm_name"].get<std::string>();
        }
        m.valid = true;
    } catch (const std::exception &e) {
        REDIS_FREE(r);
        std::cerr << "[op-metal] JSON parse error for tensor " << key << ": " << e.what() << "\n";
    }

    return m;
}

// ═══════════════════════════════════════════════════════════
// Kernel dispatch
// ═══════════════════════════════════════════════════════════

static inline int64_t element_count(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static bool dispatch_binary(const std::string &opcode, const std::string &dtype,
                            const void *a, const void *b, void *c, int64_t n) {
    using namespace deepx::metal::kernels;

    if (dtype == "f32" || dtype == "float32") {
        if (opcode == "add")      return add_f32((const float*)a, (const float*)b, (float*)c, n);
        else if (opcode == "sub") return sub_f32((const float*)a, (const float*)b, (float*)c, n);
        else if (opcode == "mul") return mul_f32((const float*)a, (const float*)b, (float*)c, n);
        else if (opcode == "div") return div_f32((const float*)a, (const float*)b, (float*)c, n);
    }
    if (dtype == "i32" || dtype == "int32") {
        if (opcode == "add")      return add_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        else if (opcode == "sub") return sub_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        else if (opcode == "mul") return mul_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
    }
    if (dtype == "i64" || dtype == "int64") {
        if (opcode == "add")      return add_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        else if (opcode == "sub") return sub_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        else if (opcode == "mul") return mul_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
    }

    return false;
}

static bool dispatch_unary(const std::string &opcode, const std::string &dtype,
                           const void *x, void *y, int64_t n) {
    using namespace deepx::metal::kernels;

    if (opcode == "relu") {
        if (dtype == "f32" || dtype == "float32") return relu_f32((const float*)x, (float*)y, n);
        if (dtype == "i32" || dtype == "int32")   return relu_i32((const int32_t*)x, (int32_t*)y, n);
        if (dtype == "i64" || dtype == "int64")   return relu_i64((const int64_t*)x, (int64_t*)y, n);
    }

    return false;
}

// ═══════════════════════════════════════════════════════════
// Task execution
// ═══════════════════════════════════════════════════════════

static void execute_task(redisContext *redis, const json &task) {
    std::string vtid   = task.value("vtid", "");
    std::string pc     = task.value("pc", "");
    std::string opcode = task.value("opcode", "");

    std::cout << "[op-metal] executing " << opcode << " vtid=" << vtid << " pc=" << pc << "\n";

    if (!task.contains("inputs") || !task.contains("outputs")) {
        notify_done(redis, vtid, pc, "error", "missing inputs/outputs");
        return;
    }

    const auto &inputs  = task["inputs"];
    const auto &outputs = task["outputs"];

    if (inputs.empty() || outputs.empty()) {
        notify_done(redis, vtid, pc, "error", "empty inputs/outputs");
        return;
    }

    // ── Resolve input tensors ──
    std::vector<TensorMeta> input_metas;
    std::vector<ShmMapping> input_shms;
    std::vector<const void*> input_ptrs;

    for (const auto &in : inputs) {
        std::string key = in.value("key", "");
        if (key.empty()) {
            notify_done(redis, vtid, pc, "error", "input missing key");
            return;
        }

        TensorMeta meta = fetch_tensor_meta(redis, key);
        if (!meta.valid) {
            notify_done(redis, vtid, pc, "error", "input tensor not found: " + key);
            return;
        }

        ShmMapping shm;
        if (!meta.shm_name.empty()) {
            if (!shm_open_readwrite(meta.shm_name, meta.byte_size, shm)) {
                notify_done(redis, vtid, pc, "error", "shm open failed: " + meta.shm_name);
                return;
            }
            input_ptrs.push_back(shm.addr);
        } else {
            notify_done(redis, vtid, pc, "error", "input has no shm address: " + key);
            return;
        }

        input_metas.push_back(meta);
        input_shms.push_back(shm);
    }

    // ── Resolve output tensor ──
    const auto &out = outputs[0];
    std::string out_key = out.value("key", "");
    TensorMeta out_meta = fetch_tensor_meta(redis, out_key);
    if (!out_meta.valid) {
        notify_done(redis, vtid, pc, "error", "output tensor not found: " + out_key);
        return;
    }

    ShmMapping out_shm;
    if (!out_meta.shm_name.empty()) {
        if (!shm_open_readwrite(out_meta.shm_name, out_meta.byte_size, out_shm)) {
            notify_done(redis, vtid, pc, "error", "output shm open failed: " + out_meta.shm_name);
            return;
        }
    } else {
        notify_done(redis, vtid, pc, "error", "output has no shm address: " + out_key);
        return;
    }

    // ── Dispatch to Metal ──
    int64_t n = element_count(out_meta.shape);
    std::string dtype = out_meta.dtype.empty() ? "f32" : out_meta.dtype;

    bool ok = false;
    if (input_ptrs.size() == 2) {
        ok = dispatch_binary(opcode, dtype, input_ptrs[0], input_ptrs[1], out_shm.addr, n);
    } else if (input_ptrs.size() == 1) {
        ok = dispatch_unary(opcode, dtype, input_ptrs[0], out_shm.addr, n);
    } else {
        notify_done(redis, vtid, pc, "error", "unsupported input count: " + std::to_string(input_ptrs.size()));
        return;
    }

    // ── Cleanup ──
    for (auto &s : input_shms) shm_close(s);
    shm_close(out_shm);

    if (ok) {
        notify_done(redis, vtid, pc, "ok");
    } else {
        notify_done(redis, vtid, pc, "error", "Metal kernel dispatch failed");
    }
}

// ═══════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    const char *redis_addr = "127.0.0.1";
    int redis_port = 6379;
    if (argc > 1) redis_addr = argv[1];
    if (argc > 2) redis_port = atoi(argv[2]);

    // 验证 Metal 可用
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "[op-metal] FATAL: no Metal device\n";
        return 1;
    }
    std::cout << "[op-metal] device: " << [[device name] UTF8String] << "\n";

    // 连接 Redis
    redisContext *redis = connect_redis(redis_addr, redis_port);
    if (!redis) return 1;
    std::cout << "[op-metal] connected to Redis " << redis_addr << ":" << redis_port << "\n";

    // 注册实例和算子
    register_instance(redis);

    std::cout << "[op-metal] listening on " << OP_QUEUE << "\n";

    // ── 消费循环 ──
    while (true) {
        redisReply *r = redis_cmd(redis, "BLPOP %s %d", OP_QUEUE, BLOCK_TIMEOUT_SEC);
        if (!r) {
            std::cerr << "[op-metal] Redis disconnected, reconnecting...\n";
            redisFree(redis);
            sleep(1);
            redis = connect_redis(redis_addr, redis_port);
            if (!redis) break;
            register_instance(redis);
            continue;
        }

        if (r->type == REDIS_REPLY_NIL) {
            REDIS_FREE(r);
            continue;
        }

        if (r->type != REDIS_REPLY_ARRAY || r->elements < 2) {
            REDIS_FREE(r);
            continue;
        }

        std::string payload(r->element[1]->str);
        REDIS_FREE(r);

        // 解析任务
        json task;
        try {
            task = json::parse(payload);
        } catch (const std::exception &e) {
            std::cerr << "[op-metal] JSON parse error: " << e.what() << "\n";
            continue;
        }

        try {
            execute_task(redis, task);
        } catch (const std::exception &e) {
            std::string vtid = task.value("vtid", "");
            std::string pc   = task.value("pc", "");
            std::cerr << "[op-metal] task exception: " << e.what() << "\n";
            if (!vtid.empty()) {
                notify_done(redis, vtid, pc, "error", e.what());
            }
        }
    }

    if (redis) {
        redis_cmd(redis, "DEL %s", INSTANCE_KEY);
        redisFree(redis);
    }
    std::cout << "[op-metal] shutdown complete.\n";
    return 0;
}
