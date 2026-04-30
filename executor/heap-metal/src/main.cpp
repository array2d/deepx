#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <unistd.h>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

#include "registry/registry_file.h"
#include "lifecycle/lifecycle.h"

using namespace deepx::heap;
using json = nlohmann::json;

static const char *HEAP_QUEUE = "cmd:heap-metal:0";
static const char *INSTANCE_KEY = "/sys/heap-plat/heap-metal:0";
static const int BLOCK_TIMEOUT_SEC = 5;

// ── Redis helpers ──

static redisContext* connect_redis(const char *addr, int port) {
    struct timeval tv = {2, 0};
    redisContext *c = redisConnectWithTimeout(addr, port, tv);
    if (!c || c->err) {
        std::cerr << "Redis connect failed: " << (c ? c->errstr : "null") << "\n";
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
    reg["program"] = "heap-metal";
    reg["device"] = "gpu0";
    reg["status"] = "running";
    reg["pid"] = getpid();
    reg["started_at"] = std::chrono::system_clock::now().time_since_epoch().count();
    redis_set(c, INSTANCE_KEY, reg.dump());
    std::cout << "[heap] registered at " << INSTANCE_KEY << "\n";
}

static void notify_done(redisContext *c, const std::string &vtid, const std::string &pc,
                         const std::string &status, const std::string &error_msg = "") {
    json done;
    done["pc"] = pc;
    done["status"] = status;
    if (!error_msg.empty()) {
        done["error"] = {{"code", "HEAP_ERROR"}, {"message", error_msg}};
    }
    std::string key = "done:" + vtid;
    redisReply *r = redis_cmd(c, "LPUSH %s %s", key.c_str(), done.dump().c_str());
    if (!r || r->type == REDIS_REPLY_ERROR) {
        std::cerr << "[heap] notify_done LPUSH failed for " << vtid << ": " << (r ? r->str : "NULL") << "\n";
    }
    REDIS_FREE(r);
    std::cout << "[heap] done " << vtid << " pc=" << pc << " status=" << status << "\n";
}

// ── JSON → LifecycleCommand ──

static LifecycleCommand parse_command(const json &j, std::string &error) {
    LifecycleCommand cmd;
    if (!j.contains("op")) {
        error = "missing 'op' field";
        return cmd;
    }
    cmd.op = j["op"].get<std::string>();

    if (j.contains("key")) {
        cmd.name = j["key"].get<std::string>();
    } else if (j.contains("src")) {
        cmd.name = j["src"].get<std::string>();
    } else {
        error = "missing tensor key";
        return cmd;
    }

    if (j.contains("dtype")) cmd.dtype = j["dtype"].get<std::string>();
    if (j.contains("shape")) {
        if (j["shape"].is_array()) {
            std::ostringstream oss;
            oss << "[";
            bool first = true;
            for (const auto &d : j["shape"]) {
                if (!first) oss << ",";
                oss << d.get<int64_t>();
                first = false;
            }
            oss << "]";
            cmd.shape = oss.str();

            int64_t total = 1;
            for (const auto &d : j["shape"]) total *= d.get<int64_t>();
            cmd.element_count = total;
        } else if (j["shape"].is_string()) {
            // shape 也可能以字符串形式传入: "[10,10]"
            cmd.shape = j["shape"].get<std::string>();
            cmd.element_count = parse_shape_size(cmd.shape);
        }
    }
    cmd.device = 0;
    cmd.pid = getpid();
    return cmd;
}

// ── Op dispatch ──

static void handle_newtensor(LifecycleManager &mgr, const LifecycleCommand &cmd,
                             redisContext *redis, const json &task) {
    std::string error;
    std::string shm_name = mgr.handle(cmd, error);

    if (!error.empty()) {
        std::cerr << "[heap] newtensor error: " << error << "\n";
        notify_done(redis, task["vtid"], task["pc"], "error", error);
        return;
    }

    // 写入 tensor 元信息到 Redis
    json meta;
    meta["dtype"] = cmd.dtype;
    meta["shape"] = json::parse(cmd.shape);
    meta["byte_size"] = cmd.element_count * 4; // default f32
    meta["device"] = "gpu0";
    meta["address"]["type"] = "shm";
    meta["address"]["shm_name"] = shm_name;
    meta["address"]["node"] = "n1";
    redis_set(redis, cmd.name, meta.dump());

    std::cout << "[heap] newtensor '" << cmd.name << "' → shm=" << shm_name << "\n";
    notify_done(redis, task["vtid"], task["pc"], "ok");
}

static void handle_deltensor(LifecycleManager &mgr, const LifecycleCommand &cmd,
                             redisContext *redis, const json &task) {
    std::string error;
    mgr.handle(cmd, error);

    if (!error.empty()) {
        std::cerr << "[heap] deltensor error: " << error << "\n";
    }

    // 删除 Redis key
    redisReply *r = redis_cmd(redis, "DEL %s", cmd.name.c_str());
    REDIS_FREE(r);

    std::cout << "[heap] deltensor '" << cmd.name << "'\n";
    notify_done(redis, task["vtid"], task["pc"], error.empty() ? "ok" : "error", error);
}

static void handle_clonetensor(LifecycleManager &mgr, const json &task, redisContext *redis) {
    std::string src = task["src"];
    std::string dst = task["dst"];

    // GET src → tensor meta, 然后在 dest 创建同名 shm 并拷贝数据
    // (简化实现: 重用 newtensor—如果不存在则创建，存在则 ref_inc)
    redisReply *r = redis_cmd(redis, "GET %s", src.c_str());
    std::string error;
    if (!r || r->type != REDIS_REPLY_STRING) {
        error = "clone: source tensor not found: " + src;
    } else {
        json src_meta = json::parse(r->str);

        LifecycleCommand cmd;
        cmd.op = "newtensor";
        cmd.name = dst;
        cmd.dtype = src_meta["dtype"];
        cmd.shape = src_meta["shape"].dump();
        cmd.device = 0;
        cmd.pid = getpid();

        int64_t total = 1;
        for (const auto &d : src_meta["shape"]) total *= d.get<int64_t>();
        cmd.element_count = total;

        std::string shm_name = mgr.handle(cmd, error);
        if (error.empty()) {
            json dst_meta = src_meta;
            dst_meta["address"]["shm_name"] = shm_name;
            redis_set(redis, dst, dst_meta.dump());

            // 拷贝数据 (如果源 shm 也在这台机器上)
            void *src_addr = mgr.get_addr(src_meta["address"]["shm_name"]);
            void *dst_addr = mgr.get_addr(shm_name);
            if (src_addr && dst_addr) {
                memcpy(dst_addr, src_addr, static_cast<size_t>(src_meta["byte_size"].get<int64_t>()));
            }
        }
    }
    REDIS_FREE(r);

    if (!error.empty()) {
        notify_done(redis, task["vtid"], task["pc"], "error", error);
    } else {
        notify_done(redis, task["vtid"], task["pc"], "ok");
    }
}

// ── Main ──

int main(int argc, char **argv) {
    const char *redis_addr = "127.0.0.1";
    int redis_port = 6379;
    if (argc > 1) redis_addr = argv[1];
    if (argc > 2) redis_port = atoi(argv[2]);

    const char *registry_path = "/tmp/deepx_heap_registry.txt";

    // 连接 Redis
    redisContext *redis = connect_redis(redis_addr, redis_port);
    if (!redis) return 1;

    std::cout << "[heap] connected to Redis " << redis_addr << ":" << redis_port << "\n";

    // 注册实例
    register_instance(redis);

    // 初始化 LifecycleManager
    FileRegistry reg(registry_path);
    LifecycleManager mgr(&reg);

    std::cout << "[heap] listening on " << HEAP_QUEUE << "\n";

    // 消费循环
    while (true) {
        redisReply *r = redis_cmd(redis, "BLPOP %s %d", HEAP_QUEUE, BLOCK_TIMEOUT_SEC);
        if (!r) {
            // Redis 断连 → 重连
            std::cerr << "[heap] Redis disconnected, reconnecting...\n";
            redisFree(redis);
            sleep(1);
            redis = connect_redis(redis_addr, redis_port);
            if (!redis) break;
            register_instance(redis);
            continue;
        }

        if (r->type == REDIS_REPLY_NIL) {
            // BLPOP timeout — no tasks
            REDIS_FREE(r);
            continue;
        }

        if (r->type != REDIS_REPLY_ARRAY || r->elements < 2) {
            REDIS_FREE(r);
            continue;
        }

        std::string payload(r->element[1]->str);
        REDIS_FREE(r);

        // 解析 JSON
        json task;
        try {
            task = json::parse(payload);
        } catch (const std::exception &e) {
            std::cerr << "[heap] JSON parse error: " << e.what() << "\n";
            continue;
        }

        std::string op = task.value("op", "");
        std::string vtid = task.value("vtid", "");
        std::string pc = task.value("pc", "");

        std::cout << "[heap] received op=" << op << " vtid=" << vtid << " pc=" << pc << "\n";

        std::string parse_err;
        LifecycleCommand cmd = parse_command(task, parse_err);

        if (!parse_err.empty()) {
            notify_done(redis, vtid, pc, "error", parse_err);
            continue;
        }

        if (op == "newtensor") {
            handle_newtensor(mgr, cmd, redis, task);
        } else if (op == "deltensor") {
            handle_deltensor(mgr, cmd, redis, task);
        } else if (op == "clonetensor") {
            handle_clonetensor(mgr, task, redis);
        } else {
            notify_done(redis, vtid, pc, "error", "unknown op: " + op);
        }
    }

    mgr.shutdown();
    if (redis) {
        redis_cmd(redis, "DEL %s", INSTANCE_KEY);
        redisFree(redis);
    }
    std::cout << "[heap] shutdown complete.\n";
    return 0;
}
