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
#include <fstream>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

#include "deepx/shm_tensor.h"

using json = nlohmann::json;

static const char *IO_QUEUE     = "cmd:io-metal:0";
static const char *SYS_QUEUE    = "sys:cmd:io-metal:0";
static const char *INSTANCE_KEY = "/sys/io-plat/io-metal:0";
static const int   BLOCK_TIMEOUT_SEC = 5;

// ═══════════════════════════════════════════════════════════
// Redis helpers
// ═══════════════════════════════════════════════════════════

static redisContext* connect_redis(const char *addr, int port) {
    struct timeval tv = {2, 0};
    redisContext *c = redisConnectWithTimeout(addr, port, tv);
    if (!c || c->err) {
        std::cerr << "[io-metal] Redis connect failed: " << (c ? c->errstr : "null") << "\n";
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
    reg["program"]    = "io-metal";
    reg["device"]     = "cpu";
    reg["status"]     = "running";
    reg["load"]       = 0.0;
    reg["pid"]        = getpid();
    reg["started_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    redis_set(c, INSTANCE_KEY, reg.dump());
    std::cout << "[io-metal] registered at " << INSTANCE_KEY << "\n";

    // ── 注册支持的 I/O 算子列表 ──
    redisReply *r = redis_cmd(c, "DEL %s", "/op/io-metal/list");
    REDIS_FREE(r);

    redis_cmd(c, "RPUSH %s %s %s %s",
              "/op/io-metal/list",
              "print", "save", "load");

    std::cout << "[io-metal] registered I/O ops: print save load\n";
}

static void notify_done(redisContext *c, const std::string &vtid,
                        const std::string &pc, const std::string &status,
                        const std::string &error_msg = "") {
    json done;
    done["pc"]     = pc;
    done["status"] = status;
    if (!error_msg.empty()) {
        done["error"] = {{"code", "IO_ERROR"}, {"message", error_msg}};
    }
    std::string key = "done:" + vtid;
    redisReply *r = redis_cmd(c, "LPUSH %s %s", key.c_str(), done.dump().c_str());
    if (!r || r->type == REDIS_REPLY_ERROR) {
        std::cerr << "[io-metal] notify_done LPUSH failed for " << vtid << ": " << (r ? r->str : "NULL") << "\n";
    }
    REDIS_FREE(r);
    std::cout << "[io-metal] done " << vtid << " pc=" << pc << " status=" << status << "\n";
}

// ═══════════════════════════════════════════════════════════
// shm helpers (reuses common-metal ShmTensor utilities)
// ═══════════════════════════════════════════════════════════

struct ShmMapping {
    std::string shm_name;
    void       *addr = nullptr;
    size_t      byte_size = 0;
};

static bool shm_open_readwrite(const std::string &name, size_t byte_size, ShmMapping &out) {
    out.shm_name  = name;
    out.byte_size = byte_size;

    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        std::cerr << "[io-metal] shm_open failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    size_t aligned = deepx::shmem::shm_page_align(byte_size);
    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::cerr << "[io-metal] mmap failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    out.addr = addr;
    return true;
}

static void shm_close(ShmMapping &m) {
    if (m.addr) {
        munmap(m.addr, deepx::shmem::shm_page_align(m.byte_size));
        m.addr = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════
// Tensor metadata from Redis
// ═══════════════════════════════════════════════════════════

struct TensorMeta {
    std::string key;
    std::string dtype;
    std::vector<int64_t> shape_data;
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
                m.shape_data.push_back(d.get<int64_t>());
            }
        }
        if (meta.contains("byte_size")) m.byte_size = meta["byte_size"].get<size_t>();
        if (meta.contains("address") && meta["address"].contains("shm_name")) {
            m.shm_name = meta["address"]["shm_name"].get<std::string>();
        }
        m.valid = true;
    } catch (const std::exception &e) {
        REDIS_FREE(r);
        std::cerr << "[io-metal] JSON parse error for tensor " << key << ": " << e.what() << "\n";
    }

    return m;
}

static inline int64_t element_count(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ═══════════════════════════════════════════════════════════
// Print helpers
// ═══════════════════════════════════════════════════════════

template <typename T>
struct type_tag { using type = T; };

#define DISPATCH_BY_DTYPE(dtype, Fn)                                         \
    do {                                                                     \
        if (dtype == "f32" || dtype == "float32")          Fn(type_tag<float>{});    \
        else if (dtype == "i64" || dtype == "int64")       Fn(type_tag<int64_t>{});  \
        else if (dtype == "i32" || dtype == "int32")       Fn(type_tag<int32_t>{});  \
        else if (dtype == "i16" || dtype == "int16")       Fn(type_tag<int16_t>{});  \
        else if (dtype == "i8"  || dtype == "int8")        Fn(type_tag<int8_t>{});   \
        else if (dtype == "bool")                           Fn(type_tag<bool>{});     \
        else { error = "unsupported dtype: " + dtype; return; }             \
    } while(0)

template <typename T>
static void io_print_data(const void *data, int64_t n, const std::string &format) {
    const T *ptr = static_cast<const T*>(data);
    std::cout << "[";
    for (int64_t i = 0; i < n; ++i) {
        if (i > 0) std::cout << " ";
        if (format.empty()) {
            std::cout << ptr[i];
        } else {
            printf(format.c_str(), ptr[i]);
        }
        if (i > 0 && i % 32 == 31 && i < n - 1) std::cout << "\n ";
    }
    std::cout << "]" << std::endl;
}

static void io_print(const std::string &dtype, const void *data, int64_t n, const std::string &format, std::string &error) {
    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        io_print_data<T>(data, n, format);
        (void)tag;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
}

// ═══════════════════════════════════════════════════════════
// I/O operations: print, save, load
// ═══════════════════════════════════════════════════════════

static size_t dtype_byte_size(const std::string &dtype) {
    if (dtype == "f64" || dtype == "float64" || dtype == "i64" || dtype == "int64") return 8;
    if (dtype == "f32" || dtype == "float32" || dtype == "i32" || dtype == "int32") return 4;
    if (dtype == "f16" || dtype == "float16" || dtype == "i16" || dtype == "int16") return 2;
    if (dtype == "i8" || dtype == "int8" || dtype == "bool") return 1;
    return 4; // default f32
}

// save: persist tensor shape + data to disk
static bool io_save(redisContext *redis, const TensorMeta &meta, const void *data,
                    const std::string &filepath, std::string &error) {
    // Write .shape as JSON
    json shape_json;
    shape_json["dtype"] = meta.dtype;
    shape_json["shape"] = meta.shape_data;
    shape_json["size"] = element_count(meta.shape_data);
    std::string shape_str = shape_json.dump();

    std::ofstream shape_fs(filepath + ".shape", std::ios::binary);
    if (!shape_fs.is_open()) {
        error = "save: cannot open " + filepath + ".shape";
        return false;
    }
    shape_fs.write(shape_str.c_str(), shape_str.size());
    shape_fs.close();

    // Write .data as raw binary
    std::ofstream data_fs(filepath + ".data", std::ios::binary);
    if (!data_fs.is_open()) {
        error = "save: cannot open " + filepath + ".data";
        return false;
    }
    data_fs.write(static_cast<const char*>(data), meta.byte_size);
    data_fs.close();

    std::cout << "[io-metal] saved tensor to " << filepath << " (dtype=" << meta.dtype
              << ", elems=" << element_count(meta.shape_data) << ")\n";
    return true;
}

// load: read tensor shape + data from disk
static bool io_load(redisContext *redis, const std::string &filepath,
                    const TensorMeta &out_meta, const std::string &out_key,
                    void *out_data, std::string &error) {
    // Read .shape
    std::ifstream shape_fs(filepath + ".shape", std::ios::binary);
    if (!shape_fs.is_open()) {
        error = "load: cannot open " + filepath + ".shape";
        return false;
    }
    std::string shape_str((std::istreambuf_iterator<char>(shape_fs)), std::istreambuf_iterator<char>());
    shape_fs.close();

    json shape_json;
    try {
        shape_json = json::parse(shape_str);
    } catch (const std::exception &e) {
        error = std::string("load: shape JSON parse error: ") + e.what();
        return false;
    }

    std::string loaded_dtype = shape_json.value("dtype", "");
    int64_t loaded_elems = shape_json.value("size", 0);
    int64_t loaded_bytes = loaded_elems * dtype_byte_size(loaded_dtype);

    // Read .data into output SHM
    std::ifstream data_fs(filepath + ".data", std::ios::binary);
    if (!data_fs.is_open()) {
        error = "load: cannot open " + filepath + ".data";
        return false;
    }

    size_t actual_read = loaded_bytes;
    if (loaded_bytes > out_meta.byte_size) {
        actual_read = out_meta.byte_size;
        std::cerr << "[io-metal] load: truncating " << loaded_bytes
                  << " → " << out_meta.byte_size << " bytes (output tensor smaller)\n";
    }
    data_fs.read(static_cast<char*>(out_data), actual_read);
    data_fs.close();

    // Update tensor metadata in Redis
    try {
        json updated_meta;
        updated_meta["dtype"] = loaded_dtype;
        updated_meta["shape"] = shape_json["shape"];
        updated_meta["byte_size"] = loaded_bytes;
        if (!out_meta.shm_name.empty()) {
            updated_meta["address"]["shm_name"] = out_meta.shm_name;
            updated_meta["address"]["node"] = "n1";
            updated_meta["address"]["type"] = "shm";
        }
        updated_meta["device"] = "cpu";
        redis_set(redis, out_key, updated_meta.dump());
    } catch (...) {
        // metadata update is best-effort
    }

    std::cout << "[io-metal] loaded tensor from " << filepath << " (dtype=" << loaded_dtype
              << ", elems=" << loaded_elems << ", bytes=" << actual_read << ")\n";
    return true;
}

// ═══════════════════════════════════════════════════════════
// Task execution
// ═══════════════════════════════════════════════════════════

static void execute_task(redisContext *redis, const json &task) {
    std::string vtid   = task.value("vtid", "");
    std::string pc     = task.value("pc", "");
    std::string opcode = task.value("opcode", "");
    json params        = task.value("params", json::object());

    if (opcode != "load" && (!task.contains("inputs") || task["inputs"].empty())) {
        notify_done(redis, vtid, pc, "error", "missing inputs for " + opcode);
        return;
    }

    // ── Resolve input tensors ──
    const auto &inputs = task.contains("inputs") ? task["inputs"] : json::array();
    std::vector<TensorMeta> input_metas;
    std::vector<ShmMapping>  input_shms;
    std::vector<void*>       input_ptrs;

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

    // ── Resolve output tensor (required for load) ──
    const auto &outputs = task.contains("outputs") ? task["outputs"] : json::array();
    std::string out_key;
    TensorMeta out_meta;
    ShmMapping out_shm;
    bool has_output = !outputs.empty();

    if (has_output) {
        const auto &out = outputs[0];
        out_key = out.value("key", "");
        out_meta = fetch_tensor_meta(redis, out_key);
        if (!out_meta.valid && opcode == "load") {
            notify_done(redis, vtid, pc, "error", "output tensor not found: " + out_key);
            for (auto &s : input_shms) shm_close(s);
            return;
        }
        if (out_meta.valid && !out_meta.shm_name.empty()) {
            if (!shm_open_readwrite(out_meta.shm_name, out_meta.byte_size, out_shm)) {
                notify_done(redis, vtid, pc, "error", "output shm open failed: " + out_meta.shm_name);
                for (auto &s : input_shms) shm_close(s);
                return;
            }
        }
    }

    std::string dtype = input_metas.empty() ? "f32" : input_metas[0].dtype;
    std::string error;
    bool ok = false;

    // ── print ──
    if (opcode == "print") {
        std::string format = params.value("format", "");
        int64_t nelem = element_count(input_metas[0].shape_data);
        io_print(dtype, input_ptrs[0], nelem, format, error);
        ok = true;
    }
    // ── save ──
    else if (opcode == "save") {
        std::string filepath = params.value("arg0", "");
        if (filepath.empty()) {
            error = "save: missing file path (arg0)";
        } else {
            ok = io_save(redis, input_metas[0], input_ptrs[0], filepath, error);
        }
    }
    // ── load ──
    else if (opcode == "load") {
        std::string filepath = params.value("arg0", "");
        if (filepath.empty()) {
            error = "load: missing file path (arg0)";
        } else if (!has_output || !out_meta.valid) {
            error = "load: missing output tensor";
        } else {
            ok = io_load(redis, filepath, out_meta, out_key, out_shm.addr, error);
        }
    }
    else {
        notify_done(redis, vtid, pc, "error",
                    "unsupported io opcode: " + opcode);
        for (auto &s : input_shms) shm_close(s);
        if (has_output) shm_close(out_shm);
        return;
    }

    // ── Cleanup ──
    for (auto &s : input_shms) shm_close(s);
    if (has_output) shm_close(out_shm);

    if (ok) {
        notify_done(redis, vtid, pc, "ok");
    } else {
        if (error.empty()) error = "io dispatch failed for " + opcode;
        notify_done(redis, vtid, pc, "error", error);
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

    // Force unbuffered output for diagnostics
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::cout << "[io-metal] I/O plane starting\n";
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) {
            std::cout << "[io-metal] CWD: " << cwd << "\n";
        }
    }

    // 连接 Redis（无限重试，不自退——io-plat 由元程控制退出）
    redisContext *redis = nullptr;
    while (!redis) {
        redis = connect_redis(redis_addr, redis_port);
        if (!redis) {
            std::cerr << "[io-metal] Redis not available, retrying in 1s...\n";
            sleep(1);
        }
    }
    std::cout << "[io-metal] connected to Redis " << redis_addr << ":" << redis_port << "\n";

    // 注册实例和算子
    register_instance(redis);

    std::cout << "[io-metal] listening on " << IO_QUEUE << " + " << SYS_QUEUE << "\n";

    // ── 消费循环 ──
    std::atomic<bool> running{true};
    while (running) {
        redisReply *r = redis_cmd(redis, "BLPOP %s %s %d", IO_QUEUE, SYS_QUEUE, BLOCK_TIMEOUT_SEC);
        if (!r) {
            // Redis 断连 → 无限重连
            std::cerr << "[io-metal] Redis disconnected, reconnecting...\n";
            redisFree(redis);
            redis = nullptr;
            while (!redis) {
                sleep(1);
                redis = connect_redis(redis_addr, redis_port);
                if (!redis) {
                    std::cerr << "[io-metal] Redis still not available, retrying...\n";
                }
            }
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

        std::string queue_name(r->element[0]->str);
        std::string payload(r->element[1]->str);
        REDIS_FREE(r);

        // ── 系统命令处理 ──
        if (queue_name == SYS_QUEUE) {
            try {
                json sys_cmd = json::parse(payload);
                std::string cmd = sys_cmd.value("cmd", "");
                if (cmd == "shutdown") {
                    std::cout << "[io-metal] received sys shutdown command, exiting...\n";
                    running = false;
                } else {
                    std::cerr << "[io-metal] unknown sys command: " << cmd << "\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "[io-metal] sys cmd JSON parse error: " << e.what() << "\n";
            }
            continue;
        }

        // ── I/O 命令处理 ──
        json task;
        try {
            task = json::parse(payload);
        } catch (const std::exception &e) {
            std::cerr << "[io-metal] JSON parse error: " << e.what() << "\n";
            continue;
        }

        try {
            execute_task(redis, task);
        } catch (const std::exception &e) {
            std::string vtid = task.value("vtid", "");
            std::string pc   = task.value("pc", "");
            std::cerr << "[io-metal] task exception: " << e.what() << "\n";
            if (!vtid.empty()) {
                notify_done(redis, vtid, pc, "error", e.what());
            }
        }
    }

    if (redis) {
        redis_cmd(redis, "DEL %s", INSTANCE_KEY);
        redisFree(redis);
    }
    std::cout << "[io-metal] shutdown complete.\n";
    return 0;
}
