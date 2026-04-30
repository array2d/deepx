# op-metal 开发约束

> op-metal 的职责边界。哪些能做，哪些**绝对不能碰**。

---

## 1. op-metal 是什么

在 DeepX 元程 5 核架构中，op-metal 是**计算平面**的 Metal GPU 实现。

它只做一件事：**被动消费指令 → 执行 GPU kernel → 通知完成**。

op-metal 是"无状态的 GPU 函数调用器"——它不关心数据从哪来、到哪去、含义是什么。

> **I/O 操作 (print/save/load) 已迁移到 `io-metal` 独立进程。** op-metal 只处理纯 GPU 计算。

---

## 2. 允许做的事（白名单）

| 操作 | 允许 | 说明 |
|------|------|------|
| BLPOP/RPOP `cmd:op-metal:*` | ✅ | 消费 VM 发来的计算指令 |
| GET Redis 获取 tensor 元信息 | ✅ | 仅限 inputs/outputs 的 dtype/shape/shm_name/byte_size |
| shm_open + mmap 映射 tensor 内存 | ✅ | 根据 shm_name 获取 CPU 指针 |
| 执行 Metal GPU kernel | ✅ | add/sub/mul/div/relu/... (不包含 I/O) |
| newBufferWithBytes/newBufferWithLength | ✅ | 将 CPU 数据拷贝到 GPU buffer |
| LPUSH `done:<vtid>` 通知完成 | ✅ | 格式: {pc, status:"ok"\|"error", error?} |
| print/save/load 等 I/O 操作 | ❌ | 已迁移到 io-metal 进程 (`cmd:io-metal:*`) |
| SET `/sys/op-plat/op-metal:0` | ✅ | 启动时注册进程状态 |
| SET NX `/op/op-metal/*` | ✅ | 首次启动时注册算子列表 |
| DELETE `/sys/op-plat/op-metal:0` | ✅ | 退出时注销 |

---

## 3. 禁止做的事（黑名单）

### 3.1 绝对禁止：理解数据语义

> op-metal 的执行单元是 **tensor 指针 + 元素数量 + dtype**。
> 它不知道也不应该知道：这个 tensor 叫什么名字、属于哪个 vthread、是不是中间结果。

| 操作 | 禁止 | 原因 |
|------|------|------|
| 检查 tensor 的内容是否正确 | ❌ | 这是 VM / 测试层的职责 |
| 打印 tensor 的采样数据 | ❌ | 数据验证不属于 op-plat |
| 比较不同 tensor 的值 | ❌ | op-plat 只做计算，不做断言 |
| 知道 tensor 的 key 名称含义 | ❌ | key 只是用来从 Redis 获取元信息 |

### 3.2 绝对禁止：格式化和展示

| 操作 | 禁止 | 原因 |
|------|------|------|
| 格式化输出结果数据（如 `[6, 8, 10, 12]`） | ❌ | 数据展示属于 VM/pysdk/deepxctl |
| 打印树形结构、表格、进度条 | ❌ | op-plat 是静默的计算后端 |
| 对执行结果做"美观"输出 | ❌ | 只有 done 通知是 op-plat 的输出 |

### 3.3 禁止：性能统计渗透到业务代码

| 操作 | 禁止 | 原因 |
|------|------|------|
| 在 execute_task 中加 chrono 计时 | ❌ | 性能统计应通过 profiling 工具（Instruments） |
| 分阶段计时（shm open / dispatch / notify） | ❌ | 同上 |
| 打印 "xxx ms" 到 stdout | ❌ | 不可与计算输出混在一起 |
| 累计统计（平均延迟、吞吐量） | ❌ | 应通过 `/sys/op-plat/` 的 metrics 字段上报 |

### 3.4 禁止：越权修改其他组件

| 操作 | 禁止 | 原因 |
|------|------|------|
| 修改 VM 的 vthread 状态 | ❌ | `/vthread/*` 是 VM 的私有空间 |
| 修改 heap-plat 的分配记录 | ❌ | heap-plat 管理 `/heap/*` |
| 消费 `done:*` 队列 | ❌ | done 是 VM 消费的 |
| 生产 `cmd:*` 队列消息 | ❌ | cmd 队列由 VM 生产 |
| 修改 build.sh / CMakeLists.txt | ❌ | 构建系统属于项目配置，非日常开发 |

### 3.5 禁止：引入调试代码

| 操作 | 禁止 | 原因 |
|------|------|------|
| fprintf(stderr, ...) 逐次调用诊断 | ❌ | 高频调用会产生大量日志 |
| std::cerr 打印每次 dispatch 的参数 | ❌ | 同上 |
| 用 `#ifdef DEBUG` 包裹大量调试输出 | ❌ | 污染主流程可读性 |
| **唯一例外**: 启动时的进程状态日志（`[op-metal] CWD/device/connected/listening`） | ✅ | 一次性输出，帮助排查进程生命周期问题 |

---

## 4. op-metal 的通信边界

```
deepxctl                 VM                    op-metal               heap-metal
   │                      │                       │                       │
   │ ── SET /vthread/1 ──→│                       │                       │
   │ ── LPUSH notify:vm ─→│                       │                       │
   │                      │── PUSH cmd:op-metal:0 ─→│                      │
   │                      │                       │── GET /data/x ────→ Redis
   │                      │                       │←── {shm_name,...} ── Redis
   │                      │                       │── shm_open("/deepx_t_xxx") → kernel
   │                      │                       │── GPU compute add_f32
   │                      │                       │── LPUSH done:1 ────→ Redis
   │                      │←─ BLPOP done:1 ────── Redis
   │                      │── PC++ 继续           │
   │←── GET /vthread/1 ───│                       │
   │      status=done      │                       │
```

**op-metal 的边界：**
- 入: `cmd:op-metal:*` 队列 + Redis GET（tensor 元信息）
- 出: `done:<vtid>` 队列
- 内部: shm 映射 → GPU 计算 → 返回

---

## 5. execute_task 的标准结构

```cpp
static void execute_task(redisContext *redis, const json &task) {
    // 1. 解析指令
    std::string opcode = task["opcode"];
    std::string vtid   = task["vtid"];
    std::string pc     = task["pc"];

    // 2. 解析 inputs → 映射 shm → 获取 GPU 指针
    std::vector<void*> input_ptrs;
    for (auto &in : task["inputs"]) {
        auto meta = fetch_tensor_meta(redis, in["key"]);
        auto shm = shm_open_readwrite(meta.shm_name, meta.byte_size);
        input_ptrs.push_back(shm.addr);
    }

    // 3. 解析 output → 映射 shm
    auto out_meta = fetch_tensor_meta(redis, task["outputs"][0]["key"]);
    auto out_shm = shm_open_readwrite(out_meta.shm_name, out_meta.byte_size);

    // 4. GPU dispatch
    bool ok = dispatch_binary(opcode, dtype, input_ptrs[0], input_ptrs[1], out_shm.addr, n);

    // 5. 清理 + 通知
    shm_close_all();
    if (ok) notify_done(redis, vtid, pc, "ok");
    else    notify_done(redis, vtid, pc, "error", "...");
}
```

**禁止在此结构中添加：**
- `auto t0 = chrono::now()` / 分阶段计时
- `std::cout << result[0:4]` / 数据采样打印
- `[op-metal] ┌─` 树形格式化输出
- 任何与计算无关的逻辑

---

## 6. 允许的日志输出

| 场景 | 允许 |
|------|------|
| 启动: 设备名、Redis 地址、监听队列 | ✅ 一次性 `std::cout` |
| 启动: CWD、进程 PID | ✅ 一次性 |
| 致命错误: Metal 设备不可用、Redis 连接失败 | ✅ `std::cerr` + `return 1` |
| 退出: shutdown complete | ✅ 一次性 |
| 每次 dispatch: opcode / dtype / n / 耗时 | ❌ 高频冗余 |
| 每次 dispatch: 结果采样数据 | ❌ 越界 |
| shm 操作成功/失败 | ❌ 成功不输出，失败走 error 通知 |

---

## 7. 与 deepxctl 的关系

| 谁做什么 | deepxctl | op-metal |
|---------|----------|----------|
| 启动 op-metal 进程 | ✅ | — |
| 注册 Metal 设备 | — | ✅ |
| 连接 Redis | — | ✅ |
| 消费计算指令 | — | ✅ |
| 执行 GPU kernel | — | ✅ |
| 验证计算结果 | ✅ (通过轮询 vthread status) | ❌ |
| 展示 tensor 数据 | ✅ (通过 deepxctl verbose) | ❌ |
| 性能分析 | ✅ (外部工具) | ❌ |
| 清理子进程 | ✅ | — |
