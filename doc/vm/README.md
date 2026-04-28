# VM 设计

> VM 是 DeepX 元程的 **解释执行引擎**，负责调度 vthread、翻译 CALL、路由指令。
> VM 是 5 核架构中的核心调度者，连接 pysdk、op-plat、heap-plat。

## 1. 定位

在元程 5 核架构中，VM 是唯一主动发起调度的组件（其他核被动消费）：

```
pysdk ──写入──→ /src/func/  +  /vthread/  +  notify:vm
                     │              │              │
                     ▼              ▼              ▼
                   Redis (KV 空间)              VM BLPOP
                     │              │              │
                     ▼              ▼              ▼
               op-plat 消费   堆变量     VM 拾取 vthread → 执行
```

| 核心 | VM 如何与之交互 |
|------|----------------|
| Redis | 全部状态存储、命令队列、通知阻塞 |
| pysdk | VM 拾取 pysdk 创建的 vthread，通过 notify:vm 唤醒 |
| op-plat | VM PUSH 计算指令到 cmd:op-*，BLPOP done:<vtid> 等待 |
| heap-plat | VM PUSH 生命周期指令到 cmd:heapmem-* |

## 2. 状态机

```
              ┌─────────┐
              │  start  │  注册 /sys/vm/<id>
              └────┬────┘
                   │
              ┌────▼────┐
         ┌───►│  idle   │◄────────────────────────┐
         │    └────┬────┘                         │
         │         │ KEYS /vthread/*               │
         │         │ 扫描 status=init             │
         │    ┌────▼────┐                         │
         │    │ running │  执行当前 vthread        │
         │    └────┬────┘                         │
         │         │                              │
         │    ┌────┴────────────────┐             │
         │    │                     │             │
         │ 计算指令              控制流            │
         │ (add,matmul...)    (call,return,if)    │
         │    │                     │             │
         │    ▼                     ▼             │
         │ PUSH→cmd:op-*       VM 直接处理         │
         │    │                eager 翻译 CALL     │
         │    ▼                DELETE 子栈 RETURN │
         │ BLPOP done:<vtid>                      │
         │    │                     │             │
         │    ▼                     │             │
         │ PC++ ◄───────────────────┘             │
         │    │                                    │
         │    │ vthread 末尾?                      │
         │    ▼                                    │
         │  status=done, GC /vthread/<vtid>/       │
         │    │                                    │
         └────┘                                    │
              │                                    │
              │ 所有 vthread done, 无新 vthread    │
         ┌────▼────┐                               │
         │  wait   │  BLPOP notify:vm              │
         │         │───────────────────────────────┘
         └─────────┘
```

## 3. 主执行循环

```
while (true) {
    // 1. 拾取 vthread
    vtid = pick_vthread();          // 原子抢占 status=init
    if (!vtid) {
        blpop("notify:vm", timeout); // 阻塞等待新 vthread
        continue;
    }

    // 2. 获取当前状态
    state = GET /vthread/<vtid>     // {pc, status, ...}
    pc = state.pc                   // "[n,0]" 或 "[n,0]/[m,0]"

    // 3. 读取当前指令
    inst = decode(vtid, pc)         // 从 [addr0,addr1] key 构建

    // 4. 执行指令
    result = execute(vtid, pc, inst)

    // 5. 根据结果推进
    if (result == "step") {
        SET /vthread/<vtid> = {pc: next(pc), status: "running"}
    }
    if (result == "wait_op") {
        SET /vthread/<vtid> = {pc: pc, status: "wait"}
        BLPOP done:<vtid>           // 阻塞等 op-plat
        SET /vthread/<vtid> = {pc: next(pc), status: "running"}
    }
    if (result == "done") {
        SET /vthread/<vtid> = {pc: pc, status: "done"}
        cleanup_vthread(vtid)       // DELETE /vthread/<vtid>/*
    }
    if (result == "error") {
        SET /vthread/<vtid> = {pc: pc, status: "error", error: ...}
    }
}
```

## 4. Vthread 拾取 (原子抢占)

多个 VM 实例并行运行时，用 Redis 事务保证每条 vthread 只被一个 VM 拾取：

```
pick_vthread():
    vtids = KEYS /vthread/*           // 扫描所有 vthread (排除 leaf sub-keys)

    for each vtid:
        WATCH /vthread/<vtid>
        val = GET /vthread/<vtid>     // {pc:"...", status:"init"}
        if val["status"] != "init":
            UNWATCH
            continue

        MULTI
        SET /vthread/<vtid> = {pc: val["pc"], status: "running"}
        result = EXEC

        if result:                    // 事务成功 → 抢占到
            return vtid

    return null                       // 没有可拾取的
```

## 5. 指令解码

从执行层坐标 `[addr0, addr1]` 读取当前指令：

```cpp
struct Instruction {
    string opcode;           // [n,0] = "add" | "call" | "return" | ...
    vector<string> reads;    // [n,-1], [n,-2], ...
    vector<string> writes;   // [n,1], [n,2], ...
    string pc;               // 当前指令坐标
};
```

**解码过程:**

```
输入: vtid="1", pc="[3,0]"
      → 基路径 = /vthread/1/[3,0]
      → MGET:
          /vthread/1/[3, 0] = "add"
          /vthread/1/[3,-1] = "./a"
          /vthread/1/[3,-2] = "./b"
          /vthread/1/[3, 1] = "./c"

输出: Instruction{opcode="add", reads=["./a", "./b"], writes=["./c"], pc="[3,0]"}
```

**子栈解码:**

```
输入: vtid="1", pc="[2,0]/[1,0]"
      → 基路径 = /vthread/1/[2,0]/[1,0]
      → 解析方式相同
```

参数中 `./` 开头的路径是命名槽位 (局部变量)，解析时会补全为 `/vthread/<vtid>/<name>`。

## 6. 指令分发

```
execute(vtid, pc, inst):
    switch inst.opcode:
    
    // === 张量计算指令 → 发给 op-plat ===
    case "add", "sub", "mul", "div", "matmul", "relu", ...:
        instance = route(inst.opcode)          // 选负载最低的 op-plat
        task = build_task(vtid, pc, inst)      // 打包 tensor 元信息+物理地址
        RPUSH cmd:op-<instance>  task
        return "wait_op"                       // 调用者 BLPOP done:<vtid>

    // === CALL → VM eager 翻译 ===
    case "call":
        func_name = inst.reads[0]              // [n,-1]
        args = inst.reads[1..]                 // [n,-2], [n,-3], ...
        ret_slot = inst.writes[0]              // [n,1]

        // 1. 确定 backend (从编译产物存在性判断)
        backend = determine_backend(func_name) // "op-metal" | "op-cuda"

        // 2. 读取编译层
        sig = GET /op/<backend>/func/<func_name>
        compiled = []:
          for i=0..:
            line = GET /op/<backend>/func/<func_name>/i
            if !line: break
            compiled.append(line)

        // 3. 形参→实参映射
        bindings = build_bindings(sig, args, ret_slot)

        // 4. 逐条 eager 翻译 → 展开为 [i,j] key
        substack = "/vthread/{vtid}/{pc}/"
        for i, dxlang_line in compiled:
            parsed = parse_dxlang(dxlang_line)
            replace_params(parsed, bindings)
            SET {substack}[{i},0] = parsed.opcode
            SET {substack}[{i},-1] = parsed.reads[0]
            ... (Pipeline 批量写)

        // 5. 进入子栈
        SET /vthread/<vtid> = {pc: pc + "/[0,0]", status: "running"}
        return "step"

    // === RETURN → 恢复父栈 ===
    case "return":
        parent_pc = pop_substack(pc)           // "[2,0]/[1,0]" → "[3,0]"
        ret_val = inst.reads[0..]               // 返回值

        // 写入父栈 CALL 指令的返回值槽位
        write_ret_to_parent(vtid, pc, ret_val)

        // DELETE 子栈
        KEYS /vthread/{vtid}/{current_substack}/* → DEL

        SET /vthread/<vtid> = {pc: parent_pc, status: "running"}
        return "step"

    // === IF → 条件跳转 ===
    case "if":
        cond = resolve(inst.reads[0])          // 条件值 (来自命名槽位)
        if cond:
            SET pc = pc + "/true/0"            // 进入 true 分支
        else:
            SET pc = pc + "/false/0"
        return "step"

    // === 生命周期指令 → 发给 heap-plat ===
    case "newtensor", "deltensor", "clonetensor":
        task = build_heap_task(vtid, pc, inst)
        RPUSH cmd:heapmem-<device> task
        BLPOP done:<vtid>                      // heap-plat 通常同步完成
        return "step"
```

## 7. CALL Eager 翻译

**为什么 eager:** CALL 时一次性将 `/op/<backend>/func/` 的 dxlang 编译层翻译为
`/vthread/` 的执行层 `[i,j]` 坐标。后续逐条执行时零解析开销——只需从本地缓存的
key-value map 读取。

**翻译过程:**

```
输入:  func_name="gemm", args=["/models/A","/models/B",1.0,0.0,"/models/C"],
       ret_slot="./out"

步骤:
  1. GET /op/op-metal/func/gemm  → "(gemm(A,B,alpha,beta,C) -> (Y))"

  2. 解析签名 → formal_params = {reads:["A","B","alpha","beta","C"], writes:["Y"]}

  3. 建立映射: {A:"/models/A", B:"/models/B", alpha:1.0, beta:0.0, C:"/models/C", Y:"./out"}

  4. 读取编译层指令 (使用 MGET 一次取回所有):
     GET /op/op-metal/func/gemm/0 = "matmul(A, B) -> ./Y"
     GET /op/op-metal/func/gemm/1 = "mul(./Y, alpha) -> ./Y"
     GET /op/op-metal/func/gemm/2 = "mul(C, beta) -> ./C"
     GET /op/op-metal/func/gemm/3 = "add(./Y, ./C) -> ./Y"

  5. 逐条翻译 + 形参替换 + 坐标展开 → Pipeline 批量写入子栈:

     翻译前 (dxlang):                          翻译后 (执行层):
     ─────────────────                        ──────────────────
     matmul(A, B) -> ./Y               →      [0, 0] = "matmul"
                                               [0,-1] = "/models/A"
                                               [0,-2] = "/models/B"
                                               [0, 1] = "./Y"

     mul(./Y, alpha) -> ./Y            →      [1, 0] = "mul"
                                               [1,-1] = "./Y"
                                               [1,-2] = "1.0"
                                               [1, 1] = "./Y"

     mul(C, beta) -> ./C               →      [2, 0] = "mul"
                                               [2,-1] = "/models/C"
                                               [2,-2] = "0.0"
                                               [2, 1] = "./C"

     add(./Y, ./C) -> ./Y              →      [3, 0] = "add"
                                               [3,-1] = "./Y"
                                               [3,-2] = "./C"
                                               [3, 1] = "./Y"

  6. SET /vthread/<vtid> = {pc: "[n,0]/[0,0]", status: "running"}
```

**翻译规则:**
- 编译层内的 `./` 相对路径 (`./Y`, `./C`) 保持 `./` 形式不变
- 编译层引用外部形参 (`A`, `B`, `C`) 替换为调用者传入的实参
- 立即数形参 (`alpha:1.0`, `beta:0.0`) 替换为具体数值
- 命名槽位 (如 `./mm`) 在 `/vthread/<vtid>/` 下平级创建，不受子栈嵌套影响

## 8. RETURN 处理

```
输入: pc="[2,0]/[1,0]" (在子栈 [2,0] 的第 1 条指令)

步骤:
  1. 解析 pc 层次:
     parent_addr = "[2,0]"      (父栈中 CALL 指令的位置)
     current_substack = "[1,0]" (当前子栈的指令位置)

  2. 读取返回值:
     // return 指令的写参数即返回值
     return_inst = decode(vtid, pc)
     ret_values = return_inst.writes   // ["./result"] 等

  3. 写入父 CALL 指令的返回值槽位:
     父 CALL 指令: /vthread/<vtid>/[2,1] = "./out"  (返回值绑定的槽位)
     将 ret_values 复制到 /vthread/<vtid>/<slot_name>

  4. DELETE 子栈:
     KEYS /vthread/<vtid>/[2,0]/* → 收集所有 key
     DEL key1 key2 ...             → 批量删除

  5. 恢复 PC:
     next_pc = "[3,0]"            (父栈 CALL 的下一条)
     SET /vthread/<vtid> = {pc: "[3,0]", status: "running"}
```

## 9. 参数解析

执行时涉及三类参数：

| 参数格式 | 解析方式 | 示例 |
|---------|---------|------|
| `./name` 相对路径 | → `/vthread/<vtid>/name` | ./mm → /vthread/1/mm |
| `/heap/...` 绝对路径 | 直接使用 | /models/W |
| 立即数 | 直接使用 | `1.0`, `true`, `"hello"` |

**Tensor 参数解析 (发给 op-plat 前):**

```
输入参数: "./a" (命名槽位)
  → GET /vthread/1/a → tensor 元信息 {dtype, shape, address:{shm_name, byte_size}, ...}
  → 打包到 op-plat 指令的 inputs[] 中
```

## 10. 多 VM 并行

```
VM-0 拾取 /vthread/1 → 执行
VM-1 拾取 /vthread/2 → 执行
VM-2 空闲 → BLPOP notify:vm

原子抢占: WATCH/MULTI/EXEC 保证同一 vthread 只被一个 VM 拾取
负载均衡: 自然竞争，先到先得（与 op-plat 的 load-based 选择不同）
```

每个 VM 实例启动时注册:

```
SET /sys/vm/0 = {"status":"running", "pid":<pid>, "started_at":<ts>}
```

## 11. 性能优化

### 11.1 批读取 (CALL 翻译)

```
当前 (逐条 GET):
  GET /op/op-metal/func/gemm/0    ← 1 次 Redis 往返
  GET /op/op-metal/func/gemm/1    ← 1 次
  GET /op/op-metal/func/gemm/2    ← 1 次
  GET /op/op-metal/func/gemm/3    ← 1 次
  → 4 次往返 (N 条指令 = N 次)

优化后 (MGET):
  MGET /op/op-metal/func/gemm/0
       /op/op-metal/func/gemm/1
       /op/op-metal/func/gemm/2
       /op/op-metal/func/gemm/3
  → 1 次往返
```

### 11.2 本地指令缓存

CALL 翻译完成后，VM 将整个子栈的 key-value 一次性读到本地内存。
后续逐条执行时零 Redis 读取，仅写 PC、PUSH 指令、BLPOP 完成通知。

```
struct SubStackCache {
    string vtid;
    string root;                            // "/vthread/1/[2,0]/"
    unordered_map<string, string> cells;    // key→value, e.g., "[0,1]"→"./Y"

    void load(RedisClient& redis) {
        auto keys = redis.keys(root + "*");
        auto vals = redis.mget(keys);
        for (int i = 0; i < keys.size(); i++)
            cells[extract_coord(keys[i])] = vals[i];
    }

    string get(int addr0, int addr1) {
        return cells[fmt("[{},{}]", addr0, addr1)];  // O(1) 本地
    }
};
```

### 11.3 批写入

```
Pipeline 批量写子栈 (1 次往返):
  Pipeline:
    SET /vthread/1/[2,0]/[0,0] = "matmul"
    SET /vthread/1/[2,0]/[0,-1] = "/models/A"
    SET /vthread/1/[2,0]/[0,1] = "./Y"
    SET /vthread/1/[2,0]/[1,0] = "mul"
    ...

  对比逐条写: N 次往返 vs 1 次
```

### 11.4 批量发射

VM 可分析指令序列，将连续无依赖的指令一次 PUSH 到 op-plat:

```
无依赖序列: [add(a,b)→c, mul(d,e)→f, relu(g)→h]
  → 三条指令输入互不依赖对方输出
  → 一次 PUSH {batch: [...]}
  → 三条全部完成后一次 LPUSH 通知
```

## 12. 错误处理

```
op-plat 返回 error:
  1. BLPOP done:<vtid> → {"status":"error", "error":{...}}
  2. SET /vthread/<vtid> = {pc:pc, status:"error", error:{...}}
  3. VM 释放该 vthread, 回到 idle

超时:
  BLPOP done:<vtid> 超时 (可配置, 默认 30s)
  → SET /vthread/<vtid> = {pc:pc, status:"error", error:{code:"TIMEOUT"}}

pysdk 可通过 GET /vthread/<vtid> 检查 status 字段感知错误。
```

## 13. 待确定问题

| 问题 | 状态 |
|------|------|
| 批量发射的依赖分析 (编译器 vs VM 运行时) | 暂定 VM 逐条发送 |
| 多 VM 负载均衡策略 | 自然竞争 (WATCH/MULTI/EXEC) |
| 本地缓存失效 (外部修改 vthread 状态) | 后续处理 |
| IF/FOR 控制流的执行层表示 | 已在 spec-v1 定义，VM 侧待实现 |
| VM 数量与 Redis 连接池 | 每 VM 一连接即可 |
