#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <functional>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static const char *PROG_NAME      = "op-cpu";
static const char *OP_QUEUE       = "cmd:op-cpu:0";
static const char *SYS_QUEUE      = "sys:cmd:op-cpu:0";
static const char *INSTANCE_KEY   = "/sys/op-plat/op-cpu:0";
static const char *HEARTBEAT_KEY  = "/sys/heartbeat/op-cpu:0";
static const int   BLOCK_TIMEOUT_SEC = 5;
static const int   HEARTBEAT_INTERVAL_SEC = 2;

// ═══════════════════════════════════════════════════════════
// Redis helpers
// ═══════════════════════════════════════════════════════════

static redisContext* connect_redis(const char *addr, int port) {
    struct timeval tv = {2, 0};
    redisContext *c = redisConnectWithTimeout(addr, port, tv);
    if (!c || c->err) {
        std::cerr << "[op-cpu] Redis connect failed: " << (c ? c->errstr : "null") << "\n";
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
    reg["program"]    = PROG_NAME;
    reg["device"]     = "cpu0";
    reg["status"]     = "running";
    reg["load"]       = 0.0;
    reg["pid"]        = getpid();
    reg["started_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    redis_set(c, INSTANCE_KEY, reg.dump());
    std::cout << "[op-cpu] registered at " << INSTANCE_KEY << "\n";

    // ── 注册支持的算子列表 ──
    redisReply *r = redis_cmd(c, "DEL %s", "/op/op-cpu/list");
    REDIS_FREE(r);

    // elementwise binary
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s",
              "/op/op-cpu/list",
              "add", "sub", "mul", "div", "max", "min", "pow");
    // elementwise unary
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s %s",
              "/op/op-cpu/list",
              "relu", "neg", "abs", "sqrt", "exp", "log", "sin", "cos", "tan");
    // elementwise scalar
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s %s %s",
              "/op/op-cpu/list",
              "addscalar", "subscalar", "mulscalar", "divscalar",
              "maxscalar", "minscalar", "powscalar",
              "rsubscalar", "rdivscalar", "rpowscalar");
    // comparison
    redis_cmd(c, "RPUSH %s %s %s %s %s %s %s %s %s",
              "/op/op-cpu/list",
              "equal", "notequal", "less", "greater",
              "equalscalar", "notequalscalar", "lessscalar", "greaterscalar");
    // changeshape
    redis_cmd(c, "RPUSH %s %s %s %s %s %s",
              "/op/op-cpu/list",
              "reshape", "transpose", "concat", "broadcastTo", "repeat");
    // reduce
    redis_cmd(c, "RPUSH %s %s %s %s %s",
              "/op/op-cpu/list",
              "sum", "prod", "reducemax", "reducemin");
    // init
    redis_cmd(c, "RPUSH %s %s %s",
              "/op/op-cpu/list",
              "constant", "arange");
    // misc
    redis_cmd(c, "RPUSH %s %s",
              "/op/op-cpu/list",
              "invert", "todtype");

    std::cout << "[op-cpu] registered all ops\n";
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
        std::cerr << "[op-cpu] notify_done LPUSH failed for " << vtid << ": " << (r ? r->str : "NULL") << "\n";
    }
    REDIS_FREE(r);
    std::cout << "[op-cpu] done " << vtid << " pc=" << pc << " status=" << status << "\n";
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
        std::cerr << "[op-cpu] shm_open failed: " << name << " (" << strerror(errno) << ")\n";
        return false;
    }

    size_t aligned = page_align(byte_size);
    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        std::cerr << "[op-cpu] mmap failed: " << name << " (" << strerror(errno) << ")\n";
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
        std::cerr << "[op-cpu] JSON parse error for tensor " << key << ": " << e.what() << "\n";
    }

    return m;
}

static inline int64_t element_count(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ═══════════════════════════════════════════════════════════
// Type dispatch
// ═══════════════════════════════════════════════════════════

template <typename T> struct type_tag { using type = T; };

#define DISPATCH_BY_DTYPE(dtype, Fn)                                         \
    do {                                                                     \
        if (dtype == "f32" || dtype == "float32")          Fn(type_tag<float>{});    \
        else if (dtype == "f64" || dtype == "float64")    Fn(type_tag<double>{});   \
        else if (dtype == "i64" || dtype == "int64")       Fn(type_tag<int64_t>{});  \
        else if (dtype == "i32" || dtype == "int32")       Fn(type_tag<int32_t>{});  \
        else if (dtype == "i16" || dtype == "int16")       Fn(type_tag<int16_t>{});  \
        else if (dtype == "i8"  || dtype == "int8")        Fn(type_tag<int8_t>{});   \
        else if (dtype == "bool")                           Fn(type_tag<bool>{});     \
        else { error = "unsupported dtype: " + dtype; return; }             \
    } while(0)

// ═══════════════════════════════════════════════════════════
// CPU kernel implementations
// ═══════════════════════════════════════════════════════════

// ── elementwise binary (CPU) ──

template <typename T>
static void cpu_add(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
template <typename T>
static void cpu_sub(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
}
template <typename T>
static void cpu_mul(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}
template <typename T>
static void cpu_div(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}
template <typename T>
static void cpu_max(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::max(a[i], b[i]);
}
template <typename T>
static void cpu_min(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::min(a[i], b[i]);
}
template <typename T>
static void cpu_pow(const T *a, const T *b, T *c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = static_cast<T>(std::pow(a[i], b[i]));
}

static bool dispatch_binary(const std::string &opcode, const std::string &dtype,
                            const void *a, const void *b, void *c, int64_t n,
                            std::string &error) {
    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T *ta = static_cast<const T*>(a);
        const T *tb = static_cast<const T*>(b);
        T *tc = static_cast<T*>(c);
        if (opcode == "add")      cpu_add<T>(ta, tb, tc, n);
        else if (opcode == "sub") cpu_sub<T>(ta, tb, tc, n);
        else if (opcode == "mul") cpu_mul<T>(ta, tb, tc, n);
        else if (opcode == "div") cpu_div<T>(ta, tb, tc, n);
        else if (opcode == "max") cpu_max<T>(ta, tb, tc, n);
        else if (opcode == "min") cpu_min<T>(ta, tb, tc, n);
        else if (opcode == "pow") cpu_pow<T>(ta, tb, tc, n);
        else error = "unknown binary op: " + opcode;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
    return error.empty();
}

// ── elementwise unary (CPU) ──

template <typename T>
static void cpu_relu(const T *x, T *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] > 0 ? x[i] : 0;
}
template <typename T>
static void cpu_neg(const T *x, T *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = -x[i];
}
template <typename T>
static void cpu_abs(const T *x, T *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = std::abs(x[i]);
}
template <typename T>
static void cpu_log(const T *x, T *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = static_cast<T>(std::log(x[i]));
}

static bool dispatch_unary(const std::string &opcode, const std::string &dtype,
                           const void *x, void *y, int64_t n,
                           std::string &error) {
    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T *tx = static_cast<const T*>(x);
        T *ty = static_cast<T*>(y);
        if (opcode == "relu") cpu_relu<T>(tx, ty, n);
        else if (opcode == "neg") cpu_neg<T>(tx, ty, n);
        else if (opcode == "abs") cpu_abs<T>(tx, ty, n);
        else if (opcode == "sqrt") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::sqrt(tx[i]));
        }
        else if (opcode == "exp") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::exp(tx[i]));
        }
        else if (opcode == "log") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::log(tx[i]));
        }
        else if (opcode == "sin") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::sin(tx[i]));
        }
        else if (opcode == "cos") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::cos(tx[i]));
        }
        else if (opcode == "tan") {
            for (int64_t i = 0; i < n; ++i) ty[i] = static_cast<T>(std::tan(tx[i]));
        }
        else error = "unknown unary op: " + opcode;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
    return error.empty();
}

// ── scalar ops (CPU) ──

template <typename T>
static void cpu_scalar_op(const std::string &opcode, const T *a, T scalar, T *c, int64_t n) {
    if (opcode == "addscalar")      { for (int64_t i = 0; i < n; ++i) c[i] = a[i] + scalar; }
    else if (opcode == "subscalar") { for (int64_t i = 0; i < n; ++i) c[i] = a[i] - scalar; }
    else if (opcode == "mulscalar") { for (int64_t i = 0; i < n; ++i) c[i] = a[i] * scalar; }
    else if (opcode == "divscalar") { for (int64_t i = 0; i < n; ++i) c[i] = a[i] / scalar; }
    else if (opcode == "maxscalar") { for (int64_t i = 0; i < n; ++i) c[i] = std::max(a[i], scalar); }
    else if (opcode == "minscalar") { for (int64_t i = 0; i < n; ++i) c[i] = std::min(a[i], scalar); }
    else if (opcode == "powscalar") { for (int64_t i = 0; i < n; ++i) c[i] = static_cast<T>(std::pow(a[i], scalar)); }
    else if (opcode == "rsubscalar") { for (int64_t i = 0; i < n; ++i) c[i] = scalar - a[i]; }
    else if (opcode == "rdivscalar") { for (int64_t i = 0; i < n; ++i) c[i] = scalar / a[i]; }
    else if (opcode == "rpowscalar") { for (int64_t i = 0; i < n; ++i) c[i] = static_cast<T>(std::pow(scalar, a[i])); }
}

// ── comparison ops (CPU) ──

template <typename T>
static void cpu_comparison(const std::string &opcode, const T *a, const T *b, bool *c, int64_t n) {
    if (opcode == "equal")       { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] == b[i]); }
    else if (opcode == "notequal") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] != b[i]); }
    else if (opcode == "less")    { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] < b[i]); }
    else if (opcode == "greater") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] > b[i]); }
}

template <typename T>
static void cpu_scalar_comparison(const std::string &opcode, const T *a, T scalar, bool *c, int64_t n) {
    if (opcode == "equalscalar")    { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] == scalar); }
    else if (opcode == "notequalscalar") { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] != scalar); }
    else if (opcode == "lessscalar")     { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] < scalar); }
    else if (opcode == "greaterscalar")  { for (int64_t i = 0; i < n; ++i) c[i] = (a[i] > scalar); }
}

// ── invert (CPU) ──

static bool dispatch_invert(const std::string &dtype, const void *x, void *y, int64_t n, std::string &error) {
    if (dtype == "i64" || dtype == "int64") {
        const int64_t *a = static_cast<const int64_t*>(x);
        int64_t *c = static_cast<int64_t*>(y);
        for (int64_t i = 0; i < n; ++i) c[i] = ~a[i];
        return true;
    }
    if (dtype == "i32" || dtype == "int32") {
        const int32_t *a = static_cast<const int32_t*>(x);
        int32_t *c = static_cast<int32_t*>(y);
        for (int64_t i = 0; i < n; ++i) c[i] = ~a[i];
        return true;
    }
    if (dtype == "i16" || dtype == "int16") {
        const int16_t *a = static_cast<const int16_t*>(x);
        int16_t *c = static_cast<int16_t*>(y);
        for (int64_t i = 0; i < n; ++i) c[i] = static_cast<int16_t>(~a[i]);
        return true;
    }
    if (dtype == "i8" || dtype == "int8") {
        const int8_t *a = static_cast<const int8_t*>(x);
        int8_t *c = static_cast<int8_t*>(y);
        for (int64_t i = 0; i < n; ++i) c[i] = ~a[i];
        return true;
    }
    if (dtype == "bool") {
        const bool *a = static_cast<const bool*>(x);
        bool *c = static_cast<bool*>(y);
        for (int64_t i = 0; i < n; ++i) c[i] = !a[i];
        return true;
    }
    error = "invert only supports integer/bool dtypes, got: " + dtype;
    return false;
}

// ── todtype (CPU) ──

static bool dispatch_todtype(const std::string &src_dtype, const std::string &dst_dtype,
                             const void *x, void *y, int64_t n, std::string &error) {
    auto cast_op = [&](auto *src_ptr, auto *dst_ptr) {
        using DstT = std::decay_t<decltype(*dst_ptr)>;
        for (int64_t i = 0; i < n; ++i) dst_ptr[i] = static_cast<DstT>(src_ptr[i]);
    };

    // f32 source
    if (src_dtype == "f32" || src_dtype == "float32") {
        const float *src = static_cast<const float*>(x);
        if (dst_dtype == "f32" || dst_dtype == "float32") cast_op(src, static_cast<float*>(y));
        else if (dst_dtype == "f64" || dst_dtype == "float64") cast_op(src, static_cast<double*>(y));
        else if (dst_dtype == "i64" || dst_dtype == "int64") cast_op(src, static_cast<int64_t*>(y));
        else if (dst_dtype == "i32" || dst_dtype == "int32") cast_op(src, static_cast<int32_t*>(y));
        else if (dst_dtype == "i16" || dst_dtype == "int16") cast_op(src, static_cast<int16_t*>(y));
        else if (dst_dtype == "i8" || dst_dtype == "int8") cast_op(src, static_cast<int8_t*>(y));
        else { error = "unsupported dst dtype: " + dst_dtype; return false; }
        return true;
    }
    // f64 source
    if (src_dtype == "f64" || src_dtype == "float64") {
        const double *src = static_cast<const double*>(x);
        if (dst_dtype == "f32" || dst_dtype == "float32") cast_op(src, static_cast<float*>(y));
        else if (dst_dtype == "f64" || dst_dtype == "float64") cast_op(src, static_cast<double*>(y));
        else if (dst_dtype == "i64" || dst_dtype == "int64") cast_op(src, static_cast<int64_t*>(y));
        else if (dst_dtype == "i32" || dst_dtype == "int32") cast_op(src, static_cast<int32_t*>(y));
        else if (dst_dtype == "i16" || dst_dtype == "int16") cast_op(src, static_cast<int16_t*>(y));
        else if (dst_dtype == "i8" || dst_dtype == "int8") cast_op(src, static_cast<int8_t*>(y));
        else { error = "unsupported dst dtype: " + dst_dtype; return false; }
        return true;
    }
    // i64 source
    if (src_dtype == "i64" || src_dtype == "int64") {
        const int64_t *src = static_cast<const int64_t*>(x);
        if (dst_dtype == "f32" || dst_dtype == "float32") cast_op(src, static_cast<float*>(y));
        else if (dst_dtype == "f64" || dst_dtype == "float64") cast_op(src, static_cast<double*>(y));
        else if (dst_dtype == "i64" || dst_dtype == "int64") cast_op(src, static_cast<int64_t*>(y));
        else if (dst_dtype == "i32" || dst_dtype == "int32") cast_op(src, static_cast<int32_t*>(y));
        else if (dst_dtype == "i16" || dst_dtype == "int16") cast_op(src, static_cast<int16_t*>(y));
        else if (dst_dtype == "i8" || dst_dtype == "int8") cast_op(src, static_cast<int8_t*>(y));
        else { error = "unsupported dst dtype: " + dst_dtype; return false; }
        return true;
    }
    // i32 source
    if (src_dtype == "i32" || src_dtype == "int32") {
        const int32_t *src = static_cast<const int32_t*>(x);
        if (dst_dtype == "f32" || dst_dtype == "float32") cast_op(src, static_cast<float*>(y));
        else if (dst_dtype == "f64" || dst_dtype == "float64") cast_op(src, static_cast<double*>(y));
        else if (dst_dtype == "i64" || dst_dtype == "int64") cast_op(src, static_cast<int64_t*>(y));
        else if (dst_dtype == "i32" || dst_dtype == "int32") cast_op(src, static_cast<int32_t*>(y));
        else if (dst_dtype == "i16" || dst_dtype == "int16") cast_op(src, static_cast<int16_t*>(y));
        else if (dst_dtype == "i8" || dst_dtype == "int8") cast_op(src, static_cast<int8_t*>(y));
        else { error = "unsupported dst dtype: " + dst_dtype; return false; }
        return true;
    }
    error = "unsupported src dtype: " + src_dtype;
    return false;
}

// ── init ops (CPU) ──

static bool dispatch_constant(const std::string &dtype, void *out, double value,
                              int64_t n, std::string &error) {
    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        T *data = static_cast<T*>(out);
        T val = static_cast<T>(value);
        for (int64_t i = 0; i < n; ++i) data[i] = val;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
    return error.empty();
}

static bool dispatch_arange(const std::string &dtype, void *out, double start, double step,
                            int64_t n, std::string &error) {
    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        T *data = static_cast<T*>(out);
        T s = static_cast<T>(start);
        T st = static_cast<T>(step);
        for (int64_t i = 0; i < n; ++i) data[i] = s + static_cast<T>(i) * st;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
    return error.empty();
}

// ── reduce ops (CPU) ──

// Convert flat index to multi-dimensional coordinates
static void flat_to_coords(int64_t flat, const std::vector<int64_t> &shape,
                           std::vector<int64_t> &coords) {
    int ndim = (int)shape.size();
    coords.resize(ndim);
    int64_t rem = flat;
    for (int d = ndim - 1; d >= 0; --d) {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
}

// Convert multi-dimensional coords to flat index
static int64_t coords_to_flat(const std::vector<int64_t> &coords,
                              const std::vector<int64_t> &shape) {
    int64_t idx = 0;
    int64_t stride = 1;
    for (int d = (int)shape.size() - 1; d >= 0; --d) {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    return idx;
}

// Compute output shape after reducing specified axes
// keepdims=true:  reduced axes become 1
// keepdims=false: reduced axes are removed
static std::vector<int64_t> reduce_output_shape(const std::vector<int64_t> &input_shape,
                                                 const std::vector<int> &axes,
                                                 bool keepdims) {
    int ndim = (int)input_shape.size();
    // Mark axes to reduce
    std::vector<bool> reduced(ndim, false);
    for (int a : axes) {
        int ax = (a < 0) ? a + ndim : a;
        if (ax >= 0 && ax < ndim) reduced[ax] = true;
    }

    std::vector<int64_t> out_shape;
    for (int d = 0; d < ndim; ++d) {
        if (reduced[d]) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(input_shape[d]);
        }
    }
    return out_shape;
}

template <typename T>
static void cpu_reduce_sum(const T *input, T *output,
                           const std::vector<int64_t> &in_shape,
                           const std::vector<int64_t> &out_shape,
                           const std::vector<bool> &reduced_axes) {
    int ndim = (int)in_shape.size();
    int64_t out_total = 1;
    for (auto d : out_shape) out_total *= d;
    if (out_total == 0) return;

    std::vector<int64_t> iter_coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
        // Map output flat index to coordinates in output space
        std::vector<int64_t> out_coords;
        flat_to_coords(out_idx, out_shape, out_coords);

        // Map output coords to input coords (skip reduced axes for iteration)
        int out_d = 0;
        for (int d = 0; d < ndim; ++d) {
            if (reduced_axes[d]) {
                iter_coords[d] = 0;
            } else {
                iter_coords[d] = out_coords[out_d];
                out_d++;
            }
        }

        // Iterate over reduced dimensions, accumulate
        T acc = 0;
        bool first = true;
        std::vector<int64_t> cur = iter_coords;

        std::function<void(int)> reduce_loop = [&](int dim) {
            if (dim == ndim) {
                int64_t in_idx = coords_to_flat(cur, in_shape);
                if (first) { acc = input[in_idx]; first = false; }
                else acc += input[in_idx];
                return;
            }
            if (reduced_axes[dim]) {
                for (int64_t k = 0; k < in_shape[dim]; ++k) {
                    cur[dim] = k;
                    reduce_loop(dim + 1);
                }
            } else {
                reduce_loop(dim + 1);
            }
        };
        reduce_loop(0);
        output[out_idx] = acc;
    }
}

template <typename T>
static void cpu_reduce_prod(const T *input, T *output,
                            const std::vector<int64_t> &in_shape,
                            const std::vector<int64_t> &out_shape,
                            const std::vector<bool> &reduced_axes) {
    int ndim = (int)in_shape.size();
    int64_t out_total = 1;
    for (auto d : out_shape) out_total *= d;
    if (out_total == 0) return;

    std::vector<int64_t> iter_coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
        std::vector<int64_t> out_coords;
        flat_to_coords(out_idx, out_shape, out_coords);

        int out_d = 0;
        for (int d = 0; d < ndim; ++d) {
            if (reduced_axes[d]) { iter_coords[d] = 0; }
            else { iter_coords[d] = out_coords[out_d]; out_d++; }
        }

        std::vector<int64_t> cur = iter_coords;
        bool first = true;
        T acc = 1;

        std::function<void(int)> reduce_loop = [&](int dim) {
            if (dim == ndim) {
                int64_t in_idx = coords_to_flat(cur, in_shape);
                if (first) { acc = input[in_idx]; first = false; }
                else acc *= input[in_idx];
                return;
            }
            if (reduced_axes[dim]) {
                for (int64_t k = 0; k < in_shape[dim]; ++k) {
                    cur[dim] = k;
                    reduce_loop(dim + 1);
                }
            } else {
                reduce_loop(dim + 1);
            }
        };
        reduce_loop(0);
        output[out_idx] = acc;
    }
}

template <typename T>
static void cpu_reduce_max(const T *input, T *output,
                           const std::vector<int64_t> &in_shape,
                           const std::vector<int64_t> &out_shape,
                           const std::vector<bool> &reduced_axes) {
    int ndim = (int)in_shape.size();
    int64_t out_total = 1;
    for (auto d : out_shape) out_total *= d;
    if (out_total == 0) return;

    std::vector<int64_t> iter_coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
        std::vector<int64_t> out_coords;
        flat_to_coords(out_idx, out_shape, out_coords);

        int out_d = 0;
        for (int d = 0; d < ndim; ++d) {
            if (reduced_axes[d]) {
                iter_coords[d] = 0;
            } else {
                iter_coords[d] = out_coords[out_d];
                out_d++;
            }
        }

        std::vector<int64_t> cur = iter_coords;
        bool first = true;
        T acc = 0;

        std::function<void(int)> reduce_loop = [&](int dim) {
            if (dim == ndim) {
                int64_t in_idx = coords_to_flat(cur, in_shape);
                if (first) { acc = input[in_idx]; first = false; }
                else acc = std::max(acc, input[in_idx]);
                return;
            }
            if (reduced_axes[dim]) {
                for (int64_t k = 0; k < in_shape[dim]; ++k) {
                    cur[dim] = k;
                    reduce_loop(dim + 1);
                }
            } else {
                reduce_loop(dim + 1);
            }
        };
        reduce_loop(0);
        output[out_idx] = acc;
    }
}

template <typename T>
static void cpu_reduce_min(const T *input, T *output,
                           const std::vector<int64_t> &in_shape,
                           const std::vector<int64_t> &out_shape,
                           const std::vector<bool> &reduced_axes) {
    int ndim = (int)in_shape.size();
    int64_t out_total = 1;
    for (auto d : out_shape) out_total *= d;
    if (out_total == 0) return;

    std::vector<int64_t> iter_coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
        std::vector<int64_t> out_coords;
        flat_to_coords(out_idx, out_shape, out_coords);

        int out_d = 0;
        for (int d = 0; d < ndim; ++d) {
            if (reduced_axes[d]) {
                iter_coords[d] = 0;
            } else {
                iter_coords[d] = out_coords[out_d];
                out_d++;
            }
        }

        std::vector<int64_t> cur = iter_coords;
        bool first = true;
        T acc = 0;

        std::function<void(int)> reduce_loop = [&](int dim) {
            if (dim == ndim) {
                int64_t in_idx = coords_to_flat(cur, in_shape);
                if (first) { acc = input[in_idx]; first = false; }
                else acc = std::min(acc, input[in_idx]);
                return;
            }
            if (reduced_axes[dim]) {
                for (int64_t k = 0; k < in_shape[dim]; ++k) {
                    cur[dim] = k;
                    reduce_loop(dim + 1);
                }
            } else {
                reduce_loop(dim + 1);
            }
        };
        reduce_loop(0);
        output[out_idx] = acc;
    }
}

static bool dispatch_reduce(const std::string &opcode, const std::string &dtype,
                            const void *input, void *output,
                            const std::vector<int64_t> &in_shape,
                            const std::vector<int> &axes, bool keepdims,
                            std::string &error) {
    int ndim = (int)in_shape.size();
    std::vector<bool> reduced_axes(ndim, false);
    for (int a : axes) {
        int ax = (a < 0) ? a + ndim : a;
        if (ax >= 0 && ax < ndim) reduced_axes[ax] = true;
    }

    auto out_shape = reduce_output_shape(in_shape, axes, keepdims);

    auto fn = [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T *in = static_cast<const T*>(input);
        T *out = static_cast<T*>(output);
        if (opcode == "sum")       cpu_reduce_sum<T>(in, out, in_shape, out_shape, reduced_axes);
        else if (opcode == "prod")      cpu_reduce_prod<T>(in, out, in_shape, out_shape, reduced_axes);
        else if (opcode == "reducemax") cpu_reduce_max<T>(in, out, in_shape, out_shape, reduced_axes);
        else if (opcode == "reducemin") cpu_reduce_min<T>(in, out, in_shape, out_shape, reduced_axes);
        else error = "unknown reduce op: " + opcode;
    };
    DISPATCH_BY_DTYPE(dtype, fn);
    return error.empty();
}

// Compute byte size for a shape with given dtype
static size_t dtype_byte_size(const std::string &dtype) {
    if (dtype == "f64" || dtype == "float64" || dtype == "i64" || dtype == "int64") return 8;
    if (dtype == "f32" || dtype == "float32" || dtype == "i32" || dtype == "int32") return 4;
    if (dtype == "i16" || dtype == "int16") return 2;
    if (dtype == "i8" || dtype == "int8" || dtype == "bool") return 1;
    return 4; // default f32
}

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
    if (opcode == "print" || opcode == "save" || opcode == "load") {
        notify_done(redis, vtid, pc, "error", "io op routed to io-metal plane — not handled by op-cpu");
        return;
    }

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

    // ── Resolve output tensor ──
    const auto &out = outputs[0];
    std::string out_key = out.value("key", "");
    TensorMeta out_meta = fetch_tensor_meta(redis, out_key);
    if (!out_meta.valid) {
        notify_done(redis, vtid, pc, "error", "output tensor not found: " + out_key);
        for (auto &s : input_shms) shm_close(s);
        return;
    }

    ShmMapping out_shm;
    if (!out_meta.shm_name.empty()) {
        if (!shm_open_readwrite(out_meta.shm_name, out_meta.byte_size, out_shm)) {
            notify_done(redis, vtid, pc, "error", "output shm open failed: " + out_meta.shm_name);
            for (auto &s : input_shms) shm_close(s);
            return;
        }
    }

    int64_t n = out_meta.valid ? element_count(out_meta.shape_data) : element_count(input_metas[0].shape_data);
    std::string dtype = out_meta.dtype.empty() ? (input_metas.empty() ? "f32" : input_metas[0].dtype) : out_meta.dtype;

    // ── Dispatch ──
    bool ok = false;
    std::string error;

    // ── elementwise binary (CPU) ──
    if (input_ptrs.size() == 2 &&
        (opcode == "add" || opcode == "sub" || opcode == "mul" ||
         opcode == "div" || opcode == "max" || opcode == "min" || opcode == "pow")) {
        ok = dispatch_binary(opcode, dtype, input_ptrs[0], input_ptrs[1], out_shm.addr, n, error);
    }
    // ── elementwise unary (CPU) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "relu" || opcode == "neg" || opcode == "abs" ||
              opcode == "sqrt" || opcode == "exp" || opcode == "log" ||
              opcode == "sin" || opcode == "cos" || opcode == "tan")) {
        ok = dispatch_unary(opcode, dtype, input_ptrs[0], out_shm.addr, n, error);
    }
    // ── elementwise scalar (CPU) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "addscalar" || opcode == "subscalar" || opcode == "mulscalar" ||
              opcode == "divscalar" || opcode == "maxscalar" || opcode == "minscalar" ||
              opcode == "powscalar" || opcode == "rsubscalar" || opcode == "rdivscalar" ||
              opcode == "rpowscalar")) {
        double scalar_val = params.value("scalar", 0.0);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            T scalar = static_cast<T>(scalar_val);
            cpu_scalar_op(opcode, static_cast<const T*>(input_ptrs[0]), scalar, static_cast<T*>(out_shm.addr), n);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── comparison (CPU) ──
    else if (input_ptrs.size() == 2 &&
             (opcode == "equal" || opcode == "notequal" ||
              opcode == "less" || opcode == "greater")) {
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            cpu_comparison(opcode, static_cast<const T*>(input_ptrs[0]),
                          static_cast<const T*>(input_ptrs[1]),
                          static_cast<bool*>(out_shm.addr), n);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── scalar comparison (CPU) ──
    else if (input_ptrs.size() == 1 &&
             (opcode == "equalscalar" || opcode == "notequalscalar" ||
              opcode == "lessscalar" || opcode == "greaterscalar")) {
        double scalar_val = params.value("scalar", 0.0);
        auto fn = [&](auto tag) {
            using T = typename decltype(tag)::type;
            T scalar = static_cast<T>(scalar_val);
            cpu_scalar_comparison(opcode, static_cast<const T*>(input_ptrs[0]),
                                 scalar, static_cast<bool*>(out_shm.addr), n);
            ok = true;
        };
        DISPATCH_BY_DTYPE(dtype, fn);
    }
    // ── invert (CPU) ──
    else if (opcode == "invert" && input_ptrs.size() == 1) {
        ok = dispatch_invert(dtype, input_ptrs[0], out_shm.addr, n, error);
    }
    // ── todtype (CPU) ──
    else if (opcode == "todtype" && input_ptrs.size() == 1) {
        std::string src_dtype = input_metas[0].dtype.empty() ? "f32" : input_metas[0].dtype;
        std::string dst_dtype = out_meta.dtype.empty() ? "f32" : out_meta.dtype;
        ok = dispatch_todtype(src_dtype, dst_dtype, input_ptrs[0], out_shm.addr, n, error);
    }
    // ── init ops (CPU) ──
    else if (opcode == "constant" && !outputs.empty()) {
        double value = params.value("value", 0.0);
        ok = dispatch_constant(dtype, out_shm.addr, value, n, error);
    }
    else if (opcode == "arange" && !outputs.empty()) {
        double start = params.value("start", 0.0);
        double step  = params.value("step", 1.0);
        ok = dispatch_arange(dtype, out_shm.addr, start, step, n, error);
    }
    // ── changeshape ops (stub — not yet rebuilt) ──
    else if (opcode == "reshape" || opcode == "transpose" || opcode == "concat" ||
             opcode == "broadcastTo" || opcode == "repeat") {
        error = "changeshape ops not available (refactoring in progress)";
    }
    // ── reduce ops (CPU) ──
    else if ((opcode == "sum" || opcode == "prod" || opcode == "reducemax" || opcode == "reducemin") &&
             input_ptrs.size() == 1) {
        // Parse axes from params
        std::vector<int> axes;
        if (params.contains("axis") && params["axis"].is_array()) {
            for (const auto &a : params["axis"]) axes.push_back(a.get<int>());
        }
        if (axes.empty()) {
            // Default: reduce all axes
            for (int d = 0; d < (int)input_metas[0].shape_data.size(); ++d) axes.push_back(d);
        }
        bool keepdims = params.value("keepdims", false);
        ok = dispatch_reduce(opcode, dtype, input_ptrs[0], out_shm.addr,
                             input_metas[0].shape_data, axes, keepdims, error);
    }
    // ── unsupported ──
    else {
        error = "unsupported opcode or input count: " + opcode + " (inputs=" + std::to_string(input_ptrs.size()) + ")";
    }

    // ── Cleanup ──
    for (auto &s : input_shms) shm_close(s);
    shm_close(out_shm);

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

    // Force unbuffered output for diagnostics
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::cout << "[op-cpu] starting CPU compute plane...\n";

    // 连接 Redis (无限重试，不自退——op-plat 由元程控制退出)
    redisContext *redis = nullptr;
    while (!redis) {
        redis = connect_redis(redis_addr, redis_port);
        if (!redis) {
            std::cerr << "[op-cpu] Redis not available, retrying in 1s...\n";
            sleep(1);
        }
    }
    std::cout << "[op-cpu] connected to Redis " << redis_addr << ":" << redis_port << "\n";

    // 注册实例和算子
    register_instance(redis);

    std::cout << "[op-cpu] listening on " << OP_QUEUE << " + " << SYS_QUEUE << "\n";
    std::cout << "[op-cpu] heartbeat → " << HEARTBEAT_KEY << " (every " << HEARTBEAT_INTERVAL_SEC << "s)\n";

    // 初始心跳
    update_heartbeat(redis, "running");

    // ── 消费循环 (同时监听业务队列和系统命令队列) ──
    std::atomic<bool> running{true};
    auto last_heartbeat = std::chrono::steady_clock::now();
    while (running) {
        redisReply *r = redis_cmd(redis, "BLPOP %s %s %d", OP_QUEUE, SYS_QUEUE, BLOCK_TIMEOUT_SEC);
        if (!r) {
            // Redis 断连 → 无限重连
            std::cerr << "[op-cpu] Redis disconnected, reconnecting...\n";
            redisFree(redis);
            redis = nullptr;
            while (!redis) {
                sleep(1);
                redis = connect_redis(redis_addr, redis_port);
                if (!redis) {
                    std::cerr << "[op-cpu] Redis still not available, retrying...\n";
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
                    std::cout << "[op-cpu] received sys shutdown command, exiting...\n";
                    running = false;
                } else {
                    std::cerr << "[op-cpu] unknown sys command: " << cmd << "\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "[op-cpu] sys cmd JSON parse error: " << e.what() << "\n";
            }
            continue;
        }

        // ── 业务命令处理 ──
        json task;
        try {
            task = json::parse(payload);
        } catch (const std::exception &e) {
            std::cerr << "[op-cpu] JSON parse error: " << e.what() << "\n";
            continue;
        }

        try {
            execute_task(redis, task);
        } catch (const std::exception &e) {
            std::string vtid = task.value("vtid", "");
            std::string pc   = task.value("pc", "");
            std::cerr << "[op-cpu] task exception: " << e.what() << "\n";
            if (!vtid.empty()) {
                notify_done(redis, vtid, pc, "error", e.what());
            }
        }
    }

    // 上报 stopped 心跳，然后注销
    if (redis) {
        update_heartbeat(redis, "stopped");
        std::cout << "[op-cpu] final heartbeat: stopped\n";
        redis_cmd(redis, "DEL %s", INSTANCE_KEY);
        redisFree(redis);
    }
    std::cout << "[op-cpu] shutdown complete.\n";
    return 0;
}
