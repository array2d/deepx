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
#include <fstream>
#include <nlohmann/json.hpp>

#include "deepx/metal_device.hpp"
#include "deepx/tensorfunc/elementwise_common.hpp"

using json = nlohmann::json;

static const char *OP_QUEUE       = "cmd:exop-metal:0";
static const char *SYS_QUEUE      = "sys:cmd:exop-metal:0";
static const char *INSTANCE_KEY   = "/sys/op-plat/exop-metal:0";
static const char *HEARTBEAT_KEY  = "/sys/heartbeat/exop-metal:0";
static const char *OP_LIST_KEY    = "/op/exop-metal/list";
static const int   BLOCK_TIMEOUT_SEC = 5;
static const int   HEARTBEAT_INTERVAL_SEC = 2;

// 动态 instance name: exop-metal-{hostname}-{pid}
static std::string g_instance_name;

static void build_instance_name() {
    // 获取 hostname
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        snprintf(hostname, sizeof(hostname), "unknown");
    }
    // 截断 hostname 第一个 '.' 之后的部分
    char *dot = strchr(hostname, '.');
    if (dot) *dot = '\0';

    // 构造 instance name: exop-metal-{hostname}-{pid}
    g_instance_name = "exop-metal-" + std::string(hostname) + "-" + std::to_string(getpid());
}

// ═══════════════════════════════════════════════════════════
// Redis helpers
// ═══════════════════════════════════════════════════════════

static redisContext* connect_redis(const char *addr, int port) {
    struct timeval tv = {2, 0};
    redisContext *c = redisConnectWithTimeout(addr, port, tv);
    if (!c || c->err) {
        std::cerr << "[exop-metal] Redis connect failed: " << (c ? c->errstr : "null") << "\n";
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

static void update_heartbeat(redisContext *c, const std::string &status) {
    json hb;
    hb["ts"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    hb["status"] = status;
    hb["pid"] = getpid();
    redis_set(c, HEARTBEAT_KEY, hb.dump());
}

static void register_instance(redisContext *c) {
    json reg;
    reg["program"]    = g_instance_name;
    reg["device"]     = "gpu0";
    reg["status"]     = "running";
    reg["load"]       = 0.0;
    reg["pid"]        = getpid();
    reg["started_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    redis_set(c, INSTANCE_KEY, reg.dump());
    std::cout << "[exop-metal] registered at " << INSTANCE_KEY << "\n";

    // ── 注册支持的算子列表 ──
    redisReply *r = redis_cmd(c, "DEL %s", OP_LIST_KEY);
    REDIS_FREE(r);

    // elementwise binary (Metal GPU)
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s",
              OP_LIST_KEY,
              "add", "sub", "mul", "div", "max", "min");
    // elementwise unary (Metal GPU)
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s %s",
              OP_LIST_KEY,
              "relu", "neg", "abs", "sqrt", "exp", "log", "sin", "cos", "tan");
    // elementwise scalar
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s",
              OP_LIST_KEY,
              "addscalar", "subscalar", "mulscalar", "divscalar",
              "maxscalar", "minscalar", "pow", "powscalar");
    // comparison
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s",
              OP_LIST_KEY,
              "equal", "notequal", "less", "greater",
              "equalscalar", "notequalscalar", "lessscalar", "greaterscalar");
    // changeshape
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s",
              OP_LIST_KEY,
              "reshape", "transpose", "concat", "broadcastTo", "indexselect", "repeat");
    // reduce
    redis_cmd(c, "RPUSH %s %s %s %s %s",
              OP_LIST_KEY,
              "sum", "prod", "reducemax", "reducemin");
    // io — migrated to io-metal (separate I/O plane)
    // init
    redis_cmd(c, "RPUSH %s %s %s",
              OP_LIST_KEY,
              "constant", "arange");
    // misc
    redis_cmd(c, "RPUSH %s %s %s",
              OP_LIST_KEY,
              "invert", "todtype");

    std::cout << "[exop-metal] registered all ops\n";
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
        std::cerr << "[exop-metal] notify_done LPUSH failed for " << vtid << ": " << (r ? r->str : "NULL") << "\n";
    }
    REDIS_FREE(r);
    std::cout << "[exop-metal] done " << vtid << " pc=" << pc << " status=" << status << "\n";
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
};

static bool shm_open_readwrite(const std::string &name, size_t byte_size, ShmMapping &out) {
    out.shm_name  = name;
    out.byte_size = byte_size;

    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        std::cerr << "[exop-metal] shm_open failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    size_t aligned = page_align(byte_size);
    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::cerr << "[exop-metal] mmap failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    out.addr = addr;
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
        std::cerr << "[exop-metal] JSON parse error for tensor " << key << ": " << e.what() << "\n";
    }

    return m;
}

// ═══════════════════════════════════════════════════════════
// Convert TensorMeta shape_data to vector<int>
// ═══════════════════════════════════════════════════════════

static std::vector<int> meta_shape(const TensorMeta &m) {
    std::vector<int> s;
    for (auto d : m.shape_data) s.push_back(static_cast<int>(d));
    return s;
}

// ═══════════════════════════════════════════════════════════
// Kernel dispatch (Metal GPU)
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
        if (opcode == "sub")      return sub_f32((const float*)a, (const float*)b, (float*)c, n);
        if (opcode == "mul")      return mul_f32((const float*)a, (const float*)b, (float*)c, n);
        if (opcode == "div")      return div_f32((const float*)a, (const float*)b, (float*)c, n);
        if (opcode == "max")      return max_f32((const float*)a, (const float*)b, (float*)c, n);
        if (opcode == "min")      return min_f32((const float*)a, (const float*)b, (float*)c, n);
    }
    if (dtype == "i8" || dtype == "int8") {
        if (opcode == "add")      return add_i8((const int8_t*)a, (const int8_t*)b, (int8_t*)c, n);
        if (opcode == "sub")      return sub_i8((const int8_t*)a, (const int8_t*)b, (int8_t*)c, n);
        if (opcode == "mul")      return mul_i8((const int8_t*)a, (const int8_t*)b, (int8_t*)c, n);
        if (opcode == "max")      return max_i8((const int8_t*)a, (const int8_t*)b, (int8_t*)c, n);
        if (opcode == "min")      return min_i8((const int8_t*)a, (const int8_t*)b, (int8_t*)c, n);
    }
    if (dtype == "i16" || dtype == "int16") {
        if (opcode == "add")      return add_i16((const int16_t*)a, (const int16_t*)b, (int16_t*)c, n);
        if (opcode == "sub")      return sub_i16((const int16_t*)a, (const int16_t*)b, (int16_t*)c, n);
        if (opcode == "mul")      return mul_i16((const int16_t*)a, (const int16_t*)b, (int16_t*)c, n);
        if (opcode == "max")      return max_i16((const int16_t*)a, (const int16_t*)b, (int16_t*)c, n);
        if (opcode == "min")      return min_i16((const int16_t*)a, (const int16_t*)b, (int16_t*)c, n);
    }
    if (dtype == "i32" || dtype == "int32") {
        if (opcode == "add")      return add_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        if (opcode == "sub")      return sub_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        if (opcode == "mul")      return mul_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        if (opcode == "max")      return max_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
        if (opcode == "min")      return min_i32((const int32_t*)a, (const int32_t*)b, (int32_t*)c, n);
    }
    if (dtype == "i64" || dtype == "int64") {
        if (opcode == "add")      return add_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        if (opcode == "sub")      return sub_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        if (opcode == "mul")      return mul_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        if (opcode == "max")      return max_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
        if (opcode == "min")      return min_i64((const int64_t*)a, (const int64_t*)b, (int64_t*)c, n);
    }

    return false;
}

static bool dispatch_unary(const std::string &opcode, const std::string &dtype,
                           const void *x, void *y, int64_t n) {
    using namespace deepx::metal::kernels;

    if (opcode == "relu") {
        if (dtype == "f32" || dtype == "float32") return relu_f32((const float*)x, (float*)y, n);
        if (dtype == "i8"  || dtype == "int8")    return relu_i8((const int8_t*)x, (int8_t*)y, n);
        if (dtype == "i16" || dtype == "int16")   return relu_i16((const int16_t*)x, (int16_t*)y, n);
        if (dtype == "i32" || dtype == "int32")   return relu_i32((const int32_t*)x, (int32_t*)y, n);
        if (dtype == "i64" || dtype == "int64")   return relu_i64((const int64_t*)x, (int64_t*)y, n);
    }
    if (opcode == "neg") {
        if (dtype == "f32" || dtype == "float32") return neg_f32((const float*)x, (float*)y, n);
        if (dtype == "i8"  || dtype == "int8")    return neg_i8((const int8_t*)x, (int8_t*)y, n);
        if (dtype == "i16" || dtype == "int16")   return neg_i16((const int16_t*)x, (int16_t*)y, n);
        if (dtype == "i32" || dtype == "int32")   return neg_i32((const int32_t*)x, (int32_t*)y, n);
        if (dtype == "i64" || dtype == "int64")   return neg_i64((const int64_t*)x, (int64_t*)y, n);
    }
    if (opcode == "abs") {
        if (dtype == "f32" || dtype == "float32") return abs_f32((const float*)x, (float*)y, n);
        if (dtype == "i8"  || dtype == "int8")    return abs_i8((const int8_t*)x, (int8_t*)y, n);
        if (dtype == "i16" || dtype == "int16")   return abs_i16((const int16_t*)x, (int16_t*)y, n);
        if (dtype == "i32" || dtype == "int32")   return abs_i32((const int32_t*)x, (int32_t*)y, n);
        if (dtype == "i64" || dtype == "int64")   return abs_i64((const int64_t*)x, (int64_t*)y, n);
    }

    // f32-only unary ops
    if (dtype == "f32" || dtype == "float32") {
        if (opcode == "sqrt") return sqrt_f32((const float*)x, (float*)y, n);
        if (opcode == "exp")  return exp_f32((const float*)x, (float*)y, n);
        if (opcode == "log")  return log_f32((const float*)x, (float*)y, n);
        if (opcode == "sin")  return sin_f32((const float*)x, (float*)y, n);
        if (opcode == "cos")  return cos_f32((const float*)x, (float*)y, n);
        if (opcode == "tan")  return tan_f32((const float*)x, (float*)y, n);
    }

    return false;
}

// ═══════════════════════════════════════════════════════════
// CPU fallback dispatch: elementwise (scalar / comparison)
// ═══════════════════════════════════════════════════════════

template <typename T>
static void cpu_binary_scalar_op(const std::string &opcode,
                                  T *a_data, T scalar, T *c_data, int64_t n) {
    if (opcode == "addscalar")  { for (int64_t i = 0; i < n; ++i) c_data[i] = a_data[i] + scalar; }
    else if (opcode == "subscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = a_data[i] - scalar; }
    else if (opcode == "mulscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = a_data[i] * scalar; }
    else if (opcode == "divscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = a_data[i] / scalar; }
    else if (opcode == "maxscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = std::max(a_data[i], scalar); }
    else if (opcode == "minscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = std::min(a_data[i], scalar); }
    else if (opcode == "powscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = std::pow(a_data[i], scalar); }
    else if (opcode == "rsubscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = scalar - a_data[i]; }
    else if (opcode == "rdivscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = scalar / a_data[i]; }
    else if (opcode == "rpowscalar") { for (int64_t i = 0; i < n; ++i) c_data[i] = std::pow(scalar, a_data[i]); }
}

template <typename T>
static void cpu_comparison_op(const std::string &opcode,
                               const T *a, const T *b, bool *c, int64_t n) {
    if (opcode == "equal")      { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] == b[i]); }
    else if (opcode == "notequal") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] != b[i]); }
    else if (opcode == "less")   { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] < b[i]); }
    else if (opcode == "greater") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] > b[i]); }
}

template <typename T>
static void cpu_scalar_comparison_op(const std::string &opcode,
                                      const T *a, T scalar, bool *c, int64_t n) {
    if (opcode == "equalscalar")    { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] == scalar); }
    else if (opcode == "notequalscalar") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] != scalar); }
    else if (opcode == "lessscalar")     { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] < scalar); }
    else if (opcode == "greaterscalar")  { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] > scalar); }
}

// ═══════════════════════════════════════════════════════════
// Type dispatch (pick correct T based on dtype string)
// Uses type tags to avoid explicit template arg syntax on lambdas.
// Caller defines: auto Fn = [&](auto tag) { using T = typename decltype(tag)::type; ... };
// ═══════════════════════════════════════════════════════════

template <typename T> struct type_tag { using type = T; };

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

// ═══════════════════════════════════════════════════════════
// Task execution
// ═══════════════════════════════════════════════════════════

static void execute_task(redisContext *redis, const json &task) {
    std::string vtid   = task.value("vtid", "");
    std::string pc     = task.value("pc", "");
    std::string opcode = task.value("opcode", "");
    json params        = task.value("params", json::object());

    if (!task.contains("inputs") || !task.contains("outputs")) {
        notify_done(redis, vtid, pc, "error", "missing inputs/outputs");
        return;
    }

    const auto &inputs  = task["inputs"];
    const auto &outputs = task["outputs"];

    // IO ops (print/save/load) routed to io-metal — not handled here
    // All remaining ops require both inputs and outputs
    if (inputs.empty() || outputs.empty()) {
        notify_done(redis, vtid, pc, "error", "empty inputs/outputs for compute op");
        return;
    }

    // ── Resolve input tensors ──
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

    // ── Resolve output tensor (optional for IO ops like print/save) ──
    std::string out_key;
    TensorMeta out_meta;
    ShmMapping out_shm;
    bool has_output = !outputs.empty();
    int64_t n = 0;
    std::string dtype = "f32";

    if (has_output) {
        const auto &out = outputs[0];
        out_key = out.value("key", "");
        out_meta = fetch_tensor_meta(redis, out_key);
        if (!out_meta.valid) {
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
        n = out_meta.valid ? element_count(out_meta.shape_data) : element_count(input_metas[0].shape_data);
        dtype = out_meta.dtype.empty() ? (input_metas.empty() ? "f32" : input_metas[0].dtype) : out_meta.dtype;
    } else {
        // For ops without outputs, infer dtype from first input
        if (!input_metas.empty()) {
            dtype = input_metas[0].dtype.empty() ? "f32" : input_metas[0].dtype;
            n = element_count(input_metas[0].shape_data);
        }
    }

    // ── Dispatch ──
    bool ok = false;
    std::string error;

    // ── elementwise binary (GPU Metal) ──
    if (input_ptrs.size() == 2 &&
        (opcode == "add" || opcode == "sub" || opcode == "mul" ||
         opcode == "div" || opcode == "max" || opcode == "min")) {
        ok = dispatch_binary(opcode, dtype, input_ptrs[0], input_ptrs[1], out_shm.addr, n);
        if (!ok) error = "Metal binary kernel dispatch failed for " + opcode + ":" + dtype;
    }
    // ── elementwise unary (GPU Metal) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "relu" || opcode == "neg" || opcode == "abs" ||
              opcode == "sqrt" || opcode == "exp" || opcode == "log" ||
              opcode == "sin" || opcode == "cos" || opcode == "tan")) {
        ok = dispatch_unary(opcode, dtype, input_ptrs[0], out_shm.addr, n);
        if (!ok) error = "Metal unary kernel dispatch failed for " + opcode + ":" + dtype;
    }
    // ── elementwise scalar (CPU) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "addscalar" || opcode == "subscalar" || opcode == "mulscalar" ||
              opcode == "divscalar" || opcode == "maxscalar" || opcode == "minscalar" ||
              opcode == "powscalar" || opcode == "rsubscalar" || opcode == "rdivscalar" ||
              opcode == "rpowscalar")) {
        double scalar_val = params.value("scalar", 0.0);
        int64_t cn = element_count(input_metas[0].shape_data);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            T scalar = static_cast<T>(scalar_val);
            cpu_binary_scalar_op(opcode, static_cast<T*>(input_ptrs[0]), scalar, static_cast<T*>(out_shm.addr), cn);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── elementwise comparison (CPU) ──
    else if (input_ptrs.size() == 2 &&
             (opcode == "equal" || opcode == "notequal" ||
              opcode == "less" || opcode == "greater")) {
        int64_t cn = element_count(input_metas[0].shape_data);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            cpu_comparison_op(opcode, static_cast<const T*>(input_ptrs[0]),
                              static_cast<const T*>(input_ptrs[1]),
                              static_cast<bool*>(out_shm.addr), cn);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── scalar comparison (CPU) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "equalscalar" || opcode == "notequalscalar" ||
              opcode == "lessscalar" || opcode == "greaterscalar")) {
        double scalar_val = params.value("scalar", 0.0);
        int64_t cn = element_count(input_metas[0].shape_data);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            T scalar = static_cast<T>(scalar_val);
            cpu_scalar_comparison_op(opcode, static_cast<const T*>(input_ptrs[0]),
                                     scalar, static_cast<bool*>(out_shm.addr), cn);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── invert (CPU, integer only) ──
    else if (opcode == "invert" && input_ptrs.size() == 1) {
        int64_t nelem = element_count(input_metas[0].shape_data);
        if (dtype == "i64" || dtype == "int64") {
            int64_t *a = static_cast<int64_t*>(input_ptrs[0]);
            int64_t *c = static_cast<int64_t*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = ~a[i];
            ok = true;
        } else if (dtype == "i32" || dtype == "int32") {
            int32_t *a = static_cast<int32_t*>(input_ptrs[0]);
            int32_t *c = static_cast<int32_t*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = ~a[i];
            ok = true;
        } else if (dtype == "i16" || dtype == "int16") {
            int16_t *a = static_cast<int16_t*>(input_ptrs[0]);
            int16_t *c = static_cast<int16_t*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = static_cast<int16_t>(~a[i]);
            ok = true;
        } else if (dtype == "i8" || dtype == "int8") {
            int8_t *a = static_cast<int8_t*>(input_ptrs[0]);
            int8_t *c = static_cast<int8_t*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = ~a[i];
            ok = true;
        } else if (dtype == "bool") {
            bool *a = static_cast<bool*>(input_ptrs[0]);
            bool *c = static_cast<bool*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = !a[i];
            ok = true;
        } else {
            error = "invert only supports integer/bool dtypes, got: " + dtype;
        }
    }
    // ── todtype (CPU) ──
    else if (opcode == "todtype" && input_ptrs.size() == 1) {
        std::string src_dtype = input_metas[0].dtype.empty() ? "f32" : input_metas[0].dtype;
        std::string dst_dtype = out_meta.dtype.empty() ? "f32" : out_meta.dtype;
        int64_t nelem = element_count(input_metas[0].shape_data);

        auto copy_data = [&](auto *src_ptr, auto *dst_ptr, int64_t count) {
            for (int64_t i = 0; i < count; ++i) dst_ptr[i] = static_cast<std::decay_t<decltype(*dst_ptr)>>(src_ptr[i]);
            ok = true;
        };

        // f32 source
        if (src_dtype == "f32" || src_dtype == "float32") {
            float *src = static_cast<float*>(input_ptrs[0]);
            if (dst_dtype == "f32" || dst_dtype == "float32")
                copy_data(src, static_cast<float*>(out_shm.addr), nelem);
            else if (dst_dtype == "i64" || dst_dtype == "int64")
                copy_data(src, static_cast<int64_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i32" || dst_dtype == "int32")
                copy_data(src, static_cast<int32_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i16" || dst_dtype == "int16")
                copy_data(src, static_cast<int16_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i8" || dst_dtype == "int8")
                copy_data(src, static_cast<int8_t*>(out_shm.addr), nelem);
            else error = "unsupported dst dtype: " + dst_dtype;
        }
        // i64 source
        else if (src_dtype == "i64" || src_dtype == "int64") {
            int64_t *src = static_cast<int64_t*>(input_ptrs[0]);
            if (dst_dtype == "f32" || dst_dtype == "float32")
                copy_data(src, static_cast<float*>(out_shm.addr), nelem);
            else if (dst_dtype == "i64" || dst_dtype == "int64")
                copy_data(src, static_cast<int64_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i32" || dst_dtype == "int32")
                copy_data(src, static_cast<int32_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i16" || dst_dtype == "int16")
                copy_data(src, static_cast<int16_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i8" || dst_dtype == "int8")
                copy_data(src, static_cast<int8_t*>(out_shm.addr), nelem);
            else error = "unsupported dst dtype: " + dst_dtype;
        }
        else if (src_dtype == "i32" || src_dtype == "int32") {
            int32_t *src = static_cast<int32_t*>(input_ptrs[0]);
            if (dst_dtype == "f32" || dst_dtype == "float32")
                copy_data(src, static_cast<float*>(out_shm.addr), nelem);
            else if (dst_dtype == "i64" || dst_dtype == "int64")
                copy_data(src, static_cast<int64_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i32" || dst_dtype == "int32")
                copy_data(src, static_cast<int32_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i16" || dst_dtype == "int16")
                copy_data(src, static_cast<int16_t*>(out_shm.addr), nelem);
            else if (dst_dtype == "i8" || dst_dtype == "int8")
                copy_data(src, static_cast<int8_t*>(out_shm.addr), nelem);
            else error = "unsupported dst dtype: " + dst_dtype;
        }
        else {
            error = "unsupported src dtype: " + src_dtype;
        }
    }
    // ── changeshape / init ops (stub — not yet rebuilt) ──
    else if (opcode == "reshape" || opcode == "transpose" || opcode == "concat" ||
             opcode == "broadcastTo" || opcode == "indexselect" || opcode == "repeat" ||
             opcode == "constant") {
        error = "changeshape/init ops not available (refactoring in progress)";
    }
    // ── reduce ops (stub — not yet rebuilt) ──
    else if (opcode == "sum" || opcode == "prod" ||
             opcode == "reducemax" || opcode == "reducemin") {
        error = "reduce ops not available (refactoring in progress)";
    }
    // ── io ops (print/save/load) — routed to io-metal plane ──
    else if (opcode == "print" || opcode == "save" || opcode == "load") {
        error = "io op routed to io-metal plane (cmd:io-metal:0) — not handled by exop-metal";
    }
    // ── pow (CPU, binary) ──
    else if (opcode == "pow" && input_ptrs.size() == 2) {
        int64_t nelem = element_count(input_metas[0].shape_data);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            T *a = static_cast<T*>(input_ptrs[0]);
            T *b = static_cast<T*>(input_ptrs[1]);
            T *c = static_cast<T*>(out_shm.addr);
            for (int64_t i = 0; i < nelem; ++i) c[i] = static_cast<T>(std::pow(a[i], b[i]));
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── unsupported ──
    else {
        notify_done(redis, vtid, pc, "error",
                    "unsupported opcode or input count: " + opcode + " (inputs=" + std::to_string(input_ptrs.size()) + ")");
        // cleanup
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
        if (error.empty()) error = "dispatch failed for " + opcode;
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

    // Force unbuffered output for diagnostics (subprocess stdout is fully buffered)
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    // 构造动态 instance name: exop-metal-{hostname}-{pid}
    build_instance_name();

    // 验证 Metal 可用 (使用 C++ wrapper)
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) {
            std::cout << "[exop-metal] CWD: " << cwd << "\n";
        }
    }
    auto deviceInfo = deepx::metal::get_default_device_info();
    if (!deviceInfo.supports_metal) {
        std::cerr << "[exop-metal] FATAL: no Metal device\n";
        return 1;
    }
    std::cout << "[exop-metal] device: " << deviceInfo.name << "\n";

    // 连接 Redis（无限重试，不自退——op-plat 由元程控制退出）
    redisContext *redis = nullptr;
    while (!redis) {
        redis = connect_redis(redis_addr, redis_port);
        if (!redis) {
            std::cerr << "[exop-metal] Redis not available, retrying in 1s...\n";
            sleep(1);
        }
    }
    std::cout << "[exop-metal] connected to Redis " << redis_addr << ":" << redis_port << "\n";

    // 注册实例和算子
    register_instance(redis);

    std::cout << "[exop-metal] listening on " << OP_QUEUE << " + " << SYS_QUEUE << "\n";
    std::cout << "[exop-metal] heartbeat → " << HEARTBEAT_KEY << " (every " << HEARTBEAT_INTERVAL_SEC << "s)\n";

    // 初始心跳
    update_heartbeat(redis, "running");

    // ── 消费循环 (同时监听业务队列和系统命令队列) ──
    std::atomic<bool> running{true};
    auto last_heartbeat = std::chrono::steady_clock::now();
    while (running) {
        redisReply *r = redis_cmd(redis, "BLPOP %s %s %d", OP_QUEUE, SYS_QUEUE, BLOCK_TIMEOUT_SEC);
        if (!r) {
            // Redis 断连 → 无限重连（不自退，op-plat 由元程控制退出）
            std::cerr << "[exop-metal] Redis disconnected, reconnecting...\n";
            redisFree(redis);
            redis = nullptr;
            while (!redis) {
                sleep(1);
                redis = connect_redis(redis_addr, redis_port);
                if (!redis) {
                    std::cerr << "[exop-metal] Redis still not available, retrying...\n";
                }
            }
            register_instance(redis);
            last_heartbeat = std::chrono::steady_clock::now();
            update_heartbeat(redis, "running");
            continue;
        }

        // ── 心跳上报 ──
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= HEARTBEAT_INTERVAL_SEC) {
            update_heartbeat(redis, "running");
            last_heartbeat = now;
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
                    std::cout << "[exop-metal] received sys shutdown command, exiting...\n";
                    running = false;
                } else {
                    std::cerr << "[exop-metal] unknown sys command: " << cmd << "\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "[exop-metal] sys cmd JSON parse error: " << e.what() << "\n";
            }
            continue;
        }

        // ── 业务命令处理 ──
        // 解析任务
        json task;
        try {
            task = json::parse(payload);
        } catch (const std::exception &e) {
            std::cerr << "[exop-metal] JSON parse error: " << e.what() << "\n";
            continue;
        }

        try {
            execute_task(redis, task);
        } catch (const std::exception &e) {
            std::string vtid = task.value("vtid", "");
            std::string pc   = task.value("pc", "");
            std::cerr << "[exop-metal] task exception: " << e.what() << "\n";
            if (!vtid.empty()) {
                notify_done(redis, vtid, pc, "error", e.what());
            }
        }
    }

    // 上报 stopped 心跳，然后注销
    if (redis) {
        update_heartbeat(redis, "stopped");
        std::cout << "[exop-metal] final heartbeat: stopped\n";
        redis_cmd(redis, "DEL %s", INSTANCE_KEY);
        redisFree(redis);
    }
    std::cout << "[exop-metal] shutdown complete.\n";
    return 0;
}
