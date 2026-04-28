# VM 开发指南

> VM 是整个元程系统的核心调度者，负责解释执行 vthread、翻译 CALL、路由指令到 op-plat。
> 此组件从零开发，是 4 个核心中工作量最大的。

## 1. 角色与职责

VM 不直接执行计算，而是将计算指令分发给 op-plat，自身处理控制流。

| 能力 | 说明 |
|------|------|
| vthread 拾取 | 扫描 `/vthread/*`，原子抢占 status=init 的 vthread |
| 指令解码 | 从执行层 `[addr0, addr1]` 坐标读取并解析指令 |
| CALL 翻译 (eager) | 读取 `/op/<backend>/func/` 编译层，翻译为执行层坐标写入子栈 |
| 指令路由 | 根据 opcode 选择支持该算子的 op-plat 实例 (负载最低) |
| 异步调度 | PUSH 指令到 op-plat/heap-plat 队列，BLPOP 等待完成 |
| RETURN 处理 | 写回返回值，递归 DELETE 子栈，恢复父 PC |
| 状态管理 | 更新 `/vthread/<vtid>` 的 pc 和 status 字段 |

## 2. 从零开始

当前 executor/ 下无 VM 目录。需全新创建。

```
executor/vm/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                  入口: Redis 连接, 主循环
│   ├── vm_loop.cpp/.h            主执行循环 (状态机)
│   ├── vthread_picker.cpp/.h     vthread 拾取 (WATCH/MULTI/EXEC)
│   ├── instruction.cpp/.h        指令数据结构 + 解码
│   ├── call_translate.cpp/.h     CALL eager 翻译 (编译层→执行层)
│   ├── return_handler.cpp/.h     RETURN 处理
│   ├── op_router.cpp/.h          算子路由 (查注册信息, 选实例)
│   └── pc_manager.cpp/.h         PC 推进、跳转、子栈管理
└── test/
    └── test_vm_loop.cpp
```

依赖: hiredis (Redis C 客户端)

## 3. VM 状态机

```
         ┌─────────┐
         │  start  │  注册到 /sys/vm/0
         └────┬────┘
              │
         ┌────▼────┐
    ┌───►│  idle   │◄──────────────────┐
    │    └────┬────┘                   │
    │         │ 扫描 /vthread/          │
    │         │ 找到 status=init       │
    │    ┌────▼────┐                   │
    │    │ running │                   │
    │    └────┬────┘                   │
    │         │                        │
    │    ┌────┴────┐                   │
    │    │         │                   │
    │ 计算指令  控制流/生命周期          │
    │    │         │                   │
    │    ▼         ▼                   │
    │  PUSH      VM 直接处理            │
    │  到op-plat  (call/if/return)     │
    │    │         │                   │
    │    ▼         │                   │
    │  BLPOP       │                   │
    │  done:<vtid> │                   │
    │    │         │                   │
    │    ▼         │                   │
    │  PC++        │                   │
    │    │         │                   │
    └────┴─────────┘                   │
         │                             │
         │ vthread done               │
    ┌────▼────┐                        │
    │  wait   │─────────────────────────┘
    │  BLPOP  │  新 vthread 创建
    │  notify:vm
    └─────────┘
```

## 4. 待开发任务

### 任务 V1: 项目骨架 + Redis 连通 + 主循环 (main.cpp)

```cpp
int main() {
    auto redis = RedisClient("127.0.0.1", 6379);

    // 注册 VM 实例
    redis.set("/sys/vm/0", json{
        {"status", "running"},
        {"pid", getpid()},
        {"started_at", time(nullptr)}
    }.dump());

    // 主循环
    while (true) {
        // 1. 扫描待执行的 vthread
        auto vtids = redis.keys("/vthread/*");
        string picked;

        for (auto& vtid : vtids) {
            if (try_pick_vthread(redis, vtid)) {
                picked = vtid;
                break;
            }
        }

        // 2. 没有找到 → 阻塞等待
        if (picked.empty()) {
            auto notification = redis.blpop("notify:vm", 5);
            if (notification) {
                picked = notification["vtid"];
                // 仍需 WATCH/MULTI/EXEC 抢占
                if (!try_pick_vthread(redis, picked)) continue;
            } else {
                continue;
            }
        }

        // 3. 执行该 vthread
        execute_vthread(redis, picked);

        // 4. 清理 done vthread
        auto status = redis.get("/vthread/" + picked);
        if (status["status"] == "done") {
            cleanup_vthread(redis, picked);
        }
    }
}
```

### 任务 V2: Vthread 原子拾取 (vthread_picker.cpp)

使用 Redis 事务避免多个 VM 实例竞争:

```cpp
bool try_pick_vthread(RedisClient& redis, string vtid) {
    string key = "/vthread/" + vtid;

    redis.watch(key);
    auto val = json::parse(redis.get(key));
    if (val["status"] != "init") {
        redis.unwatch();
        return false;
    }

    redis.multi();
    val["status"] = "running";
    redis.set(key, val.dump());
    auto results = redis.exec();

    return !results.empty();  // true = 抢占成功
}
```

### 任务 V3: 指令数据结构 + 解码 (instruction.cpp)

```cpp
struct Instruction {
    string opcode;                 // [addr0, 0] 的值
    vector<string> reads;          // [addr0, -1], [addr0, -2], ...
    vector<string> writes;         // [addr0, 1], [addr0, 2], ...
};

// 从执行层 key 解码
Instruction decode(RedisClient& redis, string vtid, string pc) {
    // pc 格式: "[n,0]" 或 "[n,0]/[m,0]" (子栈)
    string base = "/vthread/" + vtid + "/" + pc_prefix(pc);

    // MGET 批量读取该指令的所有 sub-key
    // 先用 KEYS 找所有 matching key，或按约定扫描 addr1

    Instruction inst;
    inst.opcode = redis.get(base + "/[0,0]");

    // 扫描读参数: [0,-1], [0,-2], ...
    int r = 1;
    while (true) {
        string val = redis.get(base + "/[0,-" + to_string(r) + "]");
        if (val.empty()) break;
        inst.reads.push_back(val);
        r++;
    }

    // 扫描写参数: [0,1], [0,2], ...
    int w = 1;
    while (true) {
        string val = redis.get(base + "/[0," + to_string(w) + "]");
        if (val.empty()) break;
        inst.writes.push_back(val);
        w++;
    }

    return inst;
}
```

**优化:** 实际实现应一次 MGET 获取所有 key，而非逐个 GET。见 §5 性能优化。

### 任务 V4: 指令分发 (核心)

```cpp
void execute_instruction(RedisClient& redis, string vtid, string pc, Instruction inst) {
    // === 张量计算指令 ===
    if (is_tensor_op(inst.opcode)) {
        // 1. 选择 op-plat 实例
        string instance = route_op(redis, inst.opcode);

        // 2. 解析 tensor 参数 → 获取物理地址
        json task = build_op_task(redis, vtid, pc, inst);

        // 3. PUSH 到 op-plat 命令队列
        string cmd_queue = "cmd:op-" + instance;
        redis.rpush(cmd_queue, task.dump());

        // 4. 设置等待
        set_vthread_status(redis, vtid, pc, "wait");

        // 5. 阻塞等待完成
        auto done = redis.blpop("done:" + vtid, timeout_ms);
        if (!done || done["status"] == "error") {
            set_vthread_error(redis, vtid, pc, done);
            return;
        }

        // 6. PC++
        string next_pc = pc_next(pc);
        set_vthread_status(redis, vtid, next_pc, "running");
        return;
    }

    // === CALL ===
    if (inst.opcode == "call") {
        handle_call(redis, vtid, pc, inst);
        return;
    }

    // === RETURN ===
    if (inst.opcode == "return") {
        handle_return(redis, vtid, pc);
        return;
    }

    // === IF ===
    if (inst.opcode == "if") {
        handle_if(redis, vtid, pc, inst);
        return;
    }

    // === 生命周期指令 (发到 heap-plat) ===
    if (inst.opcode == "newtensor" || inst.opcode == "deltensor") {
        redis.rpush("cmd:heapmem-metal:0", build_heap_task(vtid, pc, inst).dump());
        auto done = redis.blpop("done:" + vtid, timeout_ms);
        set_vthread_status(redis, vtid, pc_next(pc), "running");
        return;
    }
}
```

### 任务 V5: CALL 翻译 (eager) — 核心逻辑

```cpp
void handle_call(RedisClient& redis, string vtid, string pc, Instruction inst) {
    // inst.reads[0] = func_name  →  [n,-1]
    // inst.reads[1..] = args     →  [n,-2], [n,-3], ...
    // inst.writes[0] = ret_slot  →  [n, 1]

    string func_name = inst.reads[0];
    string backend = determine_backend(redis, func_name);  // "op-metal" or "op-cuda"

    // 1. 读取编译层函数签名
    string base = "/op/" + backend + "/func/" + func_name;
    auto func_sig = redis.get(base);  // "(gemm(A:tensor, ...) -> (Y:tensor))"

    // 2. 解析签名 → 形参列表
    auto formal_params = parse_signature(func_sig);
    // formal_params.reads = ["A", "B", "alpha", "beta", "C"]
    // formal_params.writes = ["Y"]

    // 3. 建立形参→实参映射
    map<string, string> bindings;
    for (int i = 0; i < formal_params.reads.size(); i++) {
        bindings[formal_params.reads[i]] = inst.reads[i + 1];  // +1 因为 reads[0] 是 func_name
    }
    for (int i = 0; i < formal_params.writes.size(); i++) {
        bindings[formal_params.writes[i]] = inst.writes[i];
    }

    // 4. 批量读取编译层所有指令
    vector<string> compiled_insts;
    for (int i = 0; ; i++) {
        string val = redis.get(base + "/" + to_string(i));
        if (val.empty()) break;
        compiled_insts.push_back(val);
    }
    // 优化: 用 MGET/Pipeline 一次读取所有指令

    // 5. 逐条翻译: 解析 dxlang → 形参替换 → 展开为执行层 [i,j] key
    string substack_root = "/vthread/" + vtid + "/" + pc + "/";
    for (int i = 0; i < compiled_insts.size(); i++) {
        // 解析 dxlang 字符串
        // "matmul(A, B) -> ./Y" → opcode="matmul", reads=["A","B"], writes=["./Y"]
        auto parsed = parse_dxlang_instruction(compiled_insts[i]);

        // 形参替换
        for (auto& r : parsed.reads) {
            if (bindings.count(r)) r = bindings[r];
        }
        for (auto& w : parsed.writes) {
            if (bindings.count(w)) w = bindings[w];
        }

        // 展开为执行层 key
        redis.set(substack_root + "[0," + to_string(i) + "]", parsed.opcode);
        for (int j = 0; j < parsed.reads.size(); j++) {
            redis.set(substack_root + "[-" + to_string(j+1) + "," + to_string(i) + "]", parsed.reads[j]);
        }
        for (int j = 0; j < parsed.writes.size(); j++) {
            redis.set(substack_root + "[" + to_string(j+1) + "," + to_string(i) + "]", parsed.writes[j]);
        }
    }
    // 优化: 用 Pipeline 批量写

    // 6. 进入子栈
    string substack_pc = pc + "/[0,0]";
    set_vthread_status(redis, vtid, substack_pc, "running");
}
```

**翻译是 eager 的:** 所有指令在 CALL 时完成翻译，执行时零解析开销。

注意: 上面代码中 `[0,i]`, `[-N,i]` 的寻址方式是按 spec-v1 规范。待 VM 编码时确认具体展开格式。

### 任务 V6: RETURN 处理

```cpp
void handle_return(RedisClient& redis, string vtid, string pc) {
    // pc 格式: "[n,0]/[m,0]/[k,0]" (可能多层嵌套)

    // 1. 找到当前子栈的父 CALL 指令
    auto [parent_pc, current_pc] = parse_substack_pc(pc);
    // parent_pc = "[n,0]"
    // current_pc = "[m,0]"

    // 2. 读取返回值 (当前子栈执行完毕的结果在命名槽位)
    //    写回到父栈 CALL 指令指定的写参数槽位
    //    (RETURN 指令的写参数即返回值)
    auto return_inst = decode(redis, vtid, pc);  // return 指令
    // ...

    // 3. 递归 DELETE 当前子栈
    string substack_path = "/vthread/" + vtid + "/" + parent_pc + "/" + current_pc + "/";
    auto keys_to_delete = redis.keys(substack_path + "*");
    redis.del(keys_to_delete);  // 批量删除

    // 4. PC 恢复到父栈 CALL 指令的下一条
    int parent_addr0 = extract_addr0(parent_pc);
    string next_pc = "[0," + to_string(parent_addr0 + 1) + "]";
    set_vthread_status(redis, vtid, next_pc, "running");
}
```

### 任务 V7: 算子路由 (op_router.cpp)

```cpp
string route_op(RedisClient& redis, string opcode) {
    // 1. 找到支持该算子的程序
    auto programs = redis.keys("/op/*/list");
    string chosen_program;

    for (auto& prog_key : programs) {
        // prog_key = "/op/op-cuda/list" or "/op/op-metal/list"
        auto list = json::parse(redis.get(prog_key));
        if (std::find(list.begin(), list.end(), opcode) != list.end()) {
            // 从 key 提取程序名: "/op/op-cuda/list" → "op-cuda"
            chosen_program = extract_program_name(prog_key);
            break;
        }
    }

    if (chosen_program.empty()) {
        throw runtime_error("no op-plat supports opcode: " + opcode);
    }

    // 2. 选择该程序下负载最低的实例
    auto instances = redis.keys("/sys/op-plat/" + chosen_program + ":*");
    float best_load = 999.0f;
    string best_instance;

    for (auto& inst_key : instances) {
        auto info = json::parse(redis.get(inst_key));
        float load = info["load"].get<float>();
        if (load < best_load) {
            best_load = load;
            best_instance = info["program"].get<string>() + ":" +
                            info["device"].get<string>();
        }
    }

    return best_instance;  // e.g., "metal:0"
}
```

### 任务 V8: 批读取优化 + 本地缓存

见 §5 性能优化。核心思路: CALL 翻译完成后，将整个子栈的 `[i,j]` key 一次性 MGET 到 VM 本地 map，后续执行走本地内存。

## 5. 性能优化

### 5.1 为什么重要

VM 每次从 Redis GET 一个 key ~100μs (网络往返)。执行 20 条指令的子栈:

```
逐条模式: GET pc + GET opcode + GET params + SET pc + PUSH + BLPOP
           → ~102 次 Redis 往返 → ~10ms

批读取 + 本地缓存: MGET 全部指令 (1次) + 仅 SET pc + PUSH + BLPOP
           → ~42 次 Redis 往返 → ~4ms  (减少 60%)
```

### 5.2 实现建议

```cpp
struct VThreadCache {
    string vtid;
    string stack_root;                          // "/vthread/1/" 或 "/vthread/1/[n,0]/"
    unordered_map<string, string> instructions; // key: "[0,0]", "[0,-1]", value: opcode/params
    json state;                                 // {pc, status}

    void load(RedisClient& redis) {
        // 1. GET state
        state = json::parse(redis.get(stack_root));

        // 2. MGET 所有指令
        auto keys = redis.keys(stack_root + "/*");
        auto vals = redis.mget(keys);
        for (int i = 0; i < keys.size(); i++) {
            instructions[keys[i]] = vals[i];
        }
    }

    string get_opcode(string pc) {
        return instructions[pc_to_opcode_key(pc)];  // 本地 O(1)
    }

    Instruction decode_local(string pc) {
        // 完全从本地 instructions map 构建, 零 Redis 访问
    }
};
```

## 6. 编译与运行 (macOS)

```bash
# 安装依赖
brew install hiredis

# 构建
cd executor/vm
mkdir -p build && cd build
cmake .. && make

# 运行 (需要先启动 heapmem-metal + op-metal)
./vm
```

## 7. 验证方法

```bash
# 集成测试 (Phase 6) — 5 个进程全部启动
# 终端1: redis-server (已运行)
# 终端2: ./heapmem_metal
# 终端3: ./op_metal
# 终端4: ./vm
# 终端5: python test_integration.py

# test_integration.py 伪代码:
# kv.set("/vthread/1/a", 1)
# kv.set("/vthread/1/b", 2)
# kv.set_func("add_test", "(add(A, B) -> (C))", ["add(A, B) -> C"])
# vtid = kv.create_vthread("add_test", {"A":"./a", "B":"./b"})
# while kv.get(f"/vthread/{vtid}")["status"] != "done": time.sleep(0.1)
# assert kv.get("/vthread/1/c") == 3
```

## 8. 开发量评估

| 任务 | 新增代码 | 难度 |
|------|---------|------|
| V1: 项目骨架 + Redis 连通 + 主循环 | ~120 行 | 低 |
| V2: Vthread 原子拾取 | ~60 行 | 中 |
| V3: 指令数据结构 + 解码 | ~150 行 | 中 |
| V4: 指令分发 (核心) | ~200 行 | 高 |
| V5: CALL eager 翻译 | ~250 行 | 高 |
| V6: RETURN 处理 | ~120 行 | 中 |
| V7: 算子路由 | ~100 行 | 中 |
| V8: 批读取 + 本地缓存 | ~200 行 | 中 |
| dxlang 解析器 | ~150 行 | 中 |
| IF/FOR 控制流 | ~150 行 | 中 |
| **合计** | **~1,500 行** | **高** |
