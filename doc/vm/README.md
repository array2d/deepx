# VM 设计

> VM 是 DeepX 元程的 **解释执行引擎**，负责调度 vthread、翻译 CALL、路由指令。
> VM 是 5 核架构中的核心调度者，连接 pysdk、op-plat、heap-plat。
> 此组件从零开发，是 4 个核心中工作量最大的。

**实现语言：Go**

VM 的本质是纯协调调度器——瓶颈在 Redis 网络 I/O，不在 CPU。
Go 的 goroutine (协程) + netpoller 模型天然适配 I/O-bound 场景：
- 一个 goroutine 阻塞在 BLPOP 上仅占 ~2KB 栈 (pthread 需 ~1MB)
- go-redis 内置连接池、Pipeline、事务，无需第三方 hiredis 封装
- 单二进制部署，交叉编译一行命令
- `encoding/json`、`net/http/pprof` 等标准库开箱即用

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
| op-plat | VM PUSH 计算指令到 cmd:op-*，BLPOP done:\<vtid\> 等待 |
| heap-plat | VM PUSH 生命周期指令到 cmd:heapmem-* |

## 2. 项目结构

```
executor/vm/
├── go.mod
├── go.sum
├── cmd/
│   └── vm/
│       └── main.go                 入口: Redis 连接, 注册 VM, 启动 worker pool
├── internal/
│   ├── engine/
│   │   ├── engine.go               主执行循环 (worker goroutine 状态机)
│   │   ├── picker.go               Vthread 原子拾取 (WATCH/MULTI/EXEC)
│   │   ├── dispatch.go             指令分发 (张量计算 / 控制流 / 生命周期)
│   │   ├── translate.go            CALL eager 翻译 + RETURN 处理
│   │   └── cache.go                子栈本地缓存
│   ├── ir/
│   │   ├── instruction.go          指令数据结构 + 解码
│   │   └── dxlang.go               Dxlang 解析器
│   ├── route/
│   │   └── router.go               算子路由 (查注册信息, 负载感知选实例)
│   └── redis/
│       └── client.go               Redis 客户端封装 (go-redis)
├── engine_test.go
└── integration_test.go
```

依赖: `github.com/redis/go-redis/v9`

```bash
cd executor/vm
go mod init git.array2d.com/ai/deepx/executor/vm
go get github.com/redis/go-redis/v9
```

## 3. 并发模型

### 3.1 Worker Pool

单进程内启动 N 个 worker goroutine (默认 `GOMAXPROCS`，可配)。
每个 worker 独立循环：拾取 vthread → 执行到底 → 再拾取。
多个 worker 通过 Redis WATCH/MULTI/EXEC 自然竞争，无需中心调度器。

```
┌──────────────────────────────────────────────────┐
│                   VM 进程                         │
│                                                   │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐     │
│  │ worker-0│  │ worker-1│  ...  │ worker-N │     │
│  │ pick()  │  │ pick()  │       │ pick()   │     │
│  │   ↓     │  │   ↓     │       │   ↓      │     │
│  │execute()│  │execute()│       │execute() │     │
│  │   ↓     │  │   ↓     │       │   ↓      │     │
│  │ pick()  │  │ BLPOP() │       │  done    │     │
│  └─────────┘  └─────────┘       └─────────┘     │
│       │             │                  │          │
│       └─────────────┼──────────────────┘          │
│                     │                             │
│              go-redis 连接池                      │
│              (PoolSize = N*2)                     │
└─────────────────────┼─────────────────────────────┘
                      │
                 ┌────▼────┐
                 │  Redis  │
                 └─────────┘
```

### 3.2 状态机

```
         ┌─────────┐
         │  start  │  注册到 /sys/vm/<id>
         └────┬────┘
              │
         ┌────▼────┐
    ┌───►│  idle   │◄──────────────────┐
    │    └────┬────┘                   │
    │         │ KEYS /vthread/*         │
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

### 3.3 连接池策略

```
PoolSize = workers × 2

每个 worker 最多同时持有 2 个连接:
  - 连接 A: 常规 GET/SET/Pipeline/MGET/WATCH/MULTI/EXEC
  - 连接 B: 阻塞 BLPOP done:<vtid>

BLPOP 会长时间占连接，因此与常规操作分池管理。
常规操作通过连接池复用，BLPOP 走独立连接。
```

## 4. 入口与 Worker 主循环

### 4.1 main.go

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "os"
    "os/signal"
    "runtime"
    "syscall"
    "time"

    "git.array2d.com/ai/deepx/executor/vm/internal/engine"
    "github.com/redis/go-redis/v9"
)

func main() {
    ctx := context.Background()

    rdb := redis.NewClient(&redis.Options{
        Addr:     "127.0.0.1:6379",
        PoolSize: runtime.GOMAXPROCS(0) * 2,
    })
    if err := rdb.Ping(ctx).Err(); err != nil {
        log.Fatalf("redis connect failed: %v", err)
    }
    defer rdb.Close()

    // 注册 VM 实例
    reg := map[string]interface{}{
        "status":     "running",
        "pid":        os.Getpid(),
        "started_at": time.Now().Unix(),
    }
    data, _ := json.Marshal(reg)
    rdb.Set(ctx, "/sys/vm/0", data, 0)

    // 启动 worker pool
    workers := runtime.GOMAXPROCS(0)
    for i := 0; i < workers; i++ {
        go engine.RunWorker(ctx, rdb, i)
    }
    log.Printf("VM started with %d workers", workers)

    // 等待信号
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig

    rdb.Del(context.Background(), "/sys/vm/0")
    log.Println("VM shutdown")
}
```

### 4.2 Worker 执行循环 (engine.go)

```go
package engine

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "git.array2d.com/ai/deepx/executor/vm/internal/ir"
    "git.array2d.com/ai/deepx/executor/vm/internal/route"
    "github.com/redis/go-redis/v9"
)

// VThreadState 存储在 /vthread/<vtid> 中
type VThreadState struct {
    PC     string `json:"pc"`
    Status string `json:"status"`
    Mode   string `json:"mode,omitempty"` // "single" | "batch", 默认 "single"
}

// RunWorker 单个 worker 的主循环
func RunWorker(ctx context.Context, rdb *redis.Client, id int) {
    log.Printf("worker-%d started", id)
    for {
        select {
        case <-ctx.Done():
            return
        default:
        }

        vtid := PickVthread(ctx, rdb)
        if vtid == "" {
            WaitForVthread(ctx, rdb)
            continue
        }

        log.Printf("worker-%d picked vthread %s", id, vtid)
        Execute(ctx, rdb, vtid)
    }
}

// Execute 执行一个 vthread 直到完成或出错
func Execute(ctx context.Context, rdb *redis.Client, vtid string) {
    for {
        state := getState(ctx, rdb, vtid)
        if state.Status == "done" || state.Status == "error" {
            return
        }

        pc := state.PC
        inst, err := ir.Decode(ctx, rdb, vtid, pc)
        if err != nil {
            setError(ctx, rdb, vtid, pc, err.Error())
            return
        }

        // 张量计算指令 → op-plat
        if ir.IsComputeOp(inst.Opcode) {
            if state.Mode == "batch" {
                // TODO: 向前扫描无依赖指令, 批量 PUSH
            }

            instance := route.Select(ctx, rdb, inst.Opcode)
            task := buildOpTask(vtid, pc, inst)
            cmdQueue := fmt.Sprintf("cmd:op-%s", instance)
            taskJSON, _ := json.Marshal(task)
            rdb.RPush(ctx, cmdQueue, taskJSON)

            setState(ctx, rdb, vtid, pc, "wait")
            done, err := waitDone(ctx, rdb, vtid, 30*time.Second)
            if err != nil || done["status"] == "error" {
                setError(ctx, rdb, vtid, pc, fmt.Sprintf("%v", done["error"]))
                return
            }
            setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
            continue
        }

        // CALL
        if inst.Opcode == "call" {
            substackPC := handleCall(ctx, rdb, vtid, pc, inst)
            setState(ctx, rdb, vtid, substackPC, "running")
            continue
        }

        // RETURN
        if inst.Opcode == "return" {
            parentPC := handleReturn(ctx, rdb, vtid, pc)
            setState(ctx, rdb, vtid, parentPC, "running")
            continue
        }

        // IF
        if inst.Opcode == "if" {
            newPC := handleIf(ctx, rdb, vtid, pc, inst)
            setState(ctx, rdb, vtid, newPC, "running")
            continue
        }

        // 生命周期指令 → heap-plat
        if ir.IsLifecycleOp(inst.Opcode) {
            task := buildHeapTask(vtid, pc, inst)
            taskJSON, _ := json.Marshal(task)
            rdb.RPush(ctx, "cmd:heapmem-metal:0", taskJSON)
            waitDone(ctx, rdb, vtid, 5*time.Second)
            setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
            continue
        }
    }
}

func getState(ctx context.Context, rdb *redis.Client, vtid string) VThreadState {
    val, err := rdb.Get(ctx, "/vthread/"+vtid).Result()
    if err != nil {
        return VThreadState{Status: "error"}
    }
    var state VThreadState
    json.Unmarshal([]byte(val), &state)
    return state
}

func setState(ctx context.Context, rdb *redis.Client, vtid string, pc, status string) {
    state := VThreadState{PC: pc, Status: status}
    data, _ := json.Marshal(state)
    rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

func setError(ctx context.Context, rdb *redis.Client, vtid string, pc string, errMsg string) {
    state := map[string]interface{}{
        "pc":     pc,
        "status": "error",
        "error":  map[string]string{"code": "VM_ERROR", "message": errMsg},
    }
    data, _ := json.Marshal(state)
    rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

func waitDone(ctx context.Context, rdb *redis.Client, vtid string, timeout time.Duration) (map[string]interface{}, error) {
    result, err := rdb.BLPop(ctx, timeout, "done:"+vtid).Result()
    if err != nil {
        return nil, fmt.Errorf("waitDone timeout: %w", err)
    }
    var done map[string]interface{}
    if len(result) > 1 {
        json.Unmarshal([]byte(result[1]), &done)
    }
    return done, nil
}

```

## 5. Vthread 拾取 (picker.go)

多个 worker 并行竞争，Redis 事务保证每条 vthread 只被一个 worker 拾取：

```go
package engine

import (
    "context"
    "encoding/json"
    "time"

    "github.com/redis/go-redis/v9"
)

// PickVthread 扫描 /vthread/*, 原子抢占 status=init 的 vthread
func PickVthread(ctx context.Context, rdb *redis.Client) string {
    keys, err := rdb.Keys(ctx, "/vthread/*").Result()
    if err != nil || len(keys) == 0 {
        return ""
    }

    for _, key := range keys {
        vtid := extractVtid(key) // "/vthread/42" → "42"
        if tryPick(ctx, rdb, vtid) {
            return vtid
        }
    }
    return ""
}

func tryPick(ctx context.Context, rdb *redis.Client, vtid string) bool {
    key := "/vthread/" + vtid

    err := rdb.Watch(ctx, func(tx *redis.Tx) error {
        val, err := tx.Get(ctx, key).Result()
        if err == redis.Nil {
            return nil
        }
        if err != nil {
            return err
        }

        var state VThreadState
        if err := json.Unmarshal([]byte(val), &state); err != nil {
            return err
        }
        if state.Status != "init" {
            return nil // 已被抢走或不在 init 状态
        }

        state.Status = "running"
        data, err := json.Marshal(state)
        if err != nil {
            return err
        }

        _, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
            pipe.Set(ctx, key, data, 0)
            return nil
        })
        return err
    }, key)

    return err == nil
}

func extractVtid(key string) string {
    const prefix = "/vthread/"
    if len(key) > len(prefix) {
        return key[len(prefix):]
    }
    return key
}

func WaitForVthread(ctx context.Context, rdb *redis.Client) {
    rdb.BLPop(ctx, 5*time.Second, "notify:vm")
}
```

## 6. 指令解码 (ir/instruction.go)

从执行层坐标 `[addr0, addr1]` 读取当前指令：

```go
package ir

import (
    "context"
    "fmt"
    "strconv"
    "strings"

    "github.com/redis/go-redis/v9"
)

// Instruction 表示执行层 [addr0, addr1] 解码后的一条指令
type Instruction struct {
    Opcode string   // [addr0, 0] = "add" | "call" | "return" | ...
    Reads  []string // [addr0, -1], [addr0, -2], ...
    Writes []string // [addr0, 1], [addr0, 2], ...
    PC     string   // 当前指令坐标, e.g., "[3,0]" 或 "[2,0]/[1,0]"
}

// Decode 从 Redis 执行层 key 解码指令
func Decode(ctx context.Context, rdb *redis.Client, vtid string, pc string) (*Instruction, error) {
    base := fmt.Sprintf("/vthread/%s/%s", vtid, pc)

    keys := []string{base + "/[0,0]"} // opcode
    for i := 1; i <= 10; i++ {
        keys = append(keys, fmt.Sprintf("%s/[0,-%d]", base, i))
        keys = append(keys, fmt.Sprintf("%s/[0,%d]", base, i))
    }

    vals, err := rdb.MGet(ctx, keys...).Result()
    if err != nil {
        return nil, err
    }

    inst := &Instruction{PC: pc}
    if s, ok := vals[0].(string); ok {
        inst.Opcode = s
    }

    for i := 1; i <= 10; i++ {
        readIdx := (i-1)*2 + 1
        writeIdx := readIdx + 1
        if readIdx < len(vals) {
            if s, ok := vals[readIdx].(string); ok && s != "" {
                inst.Reads = append(inst.Reads, s)
            }
        }
        if writeIdx < len(vals) {
            if s, ok := vals[writeIdx].(string); ok && s != "" {
                inst.Writes = append(inst.Writes, s)
            }
        }
    }

    return inst, nil
}

// DecodeFromCache 从本地缓存 map 解码 (子栈场景, 零 Redis 访问)
func DecodeFromCache(cache map[string]string, pc string) *Instruction {
    base := "/" + pc
    inst := &Instruction{PC: pc}
    inst.Opcode = cache[base+"/[0,0]"]

    for i := 1; i <= 10; i++ {
        key := fmt.Sprintf("%s/[0,-%d]", base, i)
        if v, ok := cache[key]; ok && v != "" {
            inst.Reads = append(inst.Reads, v)
        }
        key = fmt.Sprintf("%s/[0,%d]", base, i)
        if v, ok := cache[key]; ok && v != "" {
            inst.Writes = append(inst.Writes, v)
        }
    }
    return inst
}

func IsComputeOp(opcode string) bool {
    lifecycle := map[string]bool{
        "call": true, "return": true, "if": true, "for": true,
        "newtensor": true, "deltensor": true, "clonetensor": true,
    }
    return !lifecycle[opcode]
}

func IsLifecycleOp(opcode string) bool {
    return opcode == "newtensor" || opcode == "deltensor" || opcode == "clonetensor"
}

// NextPC 计算下一条指令的 PC
// "[3,0]" → "[4,0]", "[2,0]/[3,0]" → "[2,0]/[4,0]"
func NextPC(pc string) string {
    parts := strings.Split(pc, "/")
    last := parts[len(parts)-1]
    num := extractAddr0(last)
    parts[len(parts)-1] = fmt.Sprintf("[%d,0]", num+1)
    return strings.Join(parts, "/")
}

func extractAddr0(coord string) int {
    s := strings.Trim(coord, "[]")
    parts := strings.Split(s, ",")
    if len(parts) > 0 {
        n, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
        return n
    }
    return 0
}
```

**解码示例:**

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

参数中 `./` 开头的路径是命名槽位 (局部变量)，解析时补全为 `/vthread/<vtid>/<name>`。

## 7. CALL Eager 翻译 (translate.go)

**为什么 eager:** CALL 时一次性将 `/op/<backend>/func/` 的 dxlang 编译层翻译为 `/vthread/` 的执行层 `[i,j]` 坐标。后续逐条执行时零解析开销。

```go
package engine

import (
    "context"
    "fmt"
    "strings"

    "git.array2d.com/ai/deepx/executor/vm/internal/ir"
    "github.com/redis/go-redis/v9"
)

// handleCall 执行 CALL 指令的 eager 翻译，返回子栈第一条指令的 PC
func handleCall(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) string {
    funcName := inst.Reads[0]

    // 1. 确定 backend
    backend := determineBackend(ctx, rdb, funcName)

    // 2. 读取编译层函数签名
    sig, err := rdb.Get(ctx, fmt.Sprintf("/op/%s/func/%s", backend, funcName)).Result()
    if err != nil {
        sig, err = rdb.Get(ctx, "/src/func/"+funcName).Result()
        if err != nil {
            panic(fmt.Sprintf("func %s not found", funcName))
        }
    }

    // 3. 解析签名 → 形参列表
    formalParams := parseSignature(sig)

    // 4. 建立形参→实参映射
    bindings := make(map[string]string)
    for i, param := range formalParams.Reads {
        if i+1 < len(inst.Reads) {
            bindings[param] = inst.Reads[i+1]
        }
    }
    for i, param := range formalParams.Writes {
        if i < len(inst.Writes) {
            bindings[param] = inst.Writes[i]
        }
    }

    // 5. 批量 MGET 编译层所有指令
    compiled := mgetAll(ctx, rdb, fmt.Sprintf("/op/%s/func/%s", backend, funcName))

    // 6. 逐条翻译 → Pipeline 批量写入子栈
    substackRoot := fmt.Sprintf("/vthread/%s/%s/", vtid, pc)
    pipe := rdb.Pipeline()

    for i, dxlangLine := range compiled {
        parsed, err := ir.ParseDxlang(dxlangLine)
        if err != nil {
            continue
        }

        replaceParams(parsed.Reads, bindings)
        replaceParams(parsed.Writes, bindings)

        pipe.Set(ctx, fmt.Sprintf("%s[%d,0]", substackRoot, i), parsed.Opcode, 0)
        for j, r := range parsed.Reads {
            pipe.Set(ctx, fmt.Sprintf("%s[%d,-%d]", substackRoot, i, j+1), r, 0)
        }
        for j, w := range parsed.Writes {
            pipe.Set(ctx, fmt.Sprintf("%s[%d,%d]", substackRoot, i, j+1), w, 0)
        }
    }

    _, err = pipe.Exec(ctx)
    if err != nil {
        panic(fmt.Sprintf("CALL translate pipeline failed: %v", err))
    }

    return pc + "/[0,0]"
}

func determineBackend(ctx context.Context, rdb *redis.Client, funcName string) string {
    for _, b := range []string{"op-metal", "op-cuda", "op-cpu"} {
        exists, _ := rdb.Exists(ctx, fmt.Sprintf("/op/%s/func/%s", b, funcName)).Result()
        if exists > 0 {
            return b
        }
    }
    return "op-metal"
}

func mgetAll(ctx context.Context, rdb *redis.Client, base string) []string {
    keys, _ := rdb.Keys(ctx, base+"/*").Result()
    if len(keys) == 0 {
        return nil
    }
    vals, _ := rdb.MGet(ctx, keys...).Result()
    result := make([]string, 0, len(vals))
    for _, v := range vals {
        if s, ok := v.(string); ok {
            result = append(result, s)
        }
    }
    return result
}

func replaceParams(params []string, bindings map[string]string) {
    for i, p := range params {
        if v, ok := bindings[p]; ok {
            params[i] = v
        }
    }
}
```

**翻译示例 (gemm):**

```
输入: func_name="gemm", args=["/models/A","/models/B",1.0,0.0,"/models/C"], ret_slot="./out"

编译层 → 执行层:
  matmul(A, B) -> ./Y       →  [0,0]="matmul", [0,-1]="/models/A", [0,-2]="/models/B", [0,1]="./Y"
  mul(./Y, alpha) -> ./Y    →  [1,0]="mul",    [1,-1]="./Y",      [1,-2]="1.0",        [1,1]="./Y"
  mul(C, beta) -> ./C       →  [2,0]="mul",    [2,-1]="/models/C", [2,-2]="0.0",        [2,1]="./C"
  add(./Y, ./C) -> ./Y      →  [3,0]="add",    [3,-1]="./Y",      [3,-2]="./C",        [3,1]="./Y"

一次 Pipeline Exec 写入 16 个 key，1 次 Redis 往返。
```

**翻译规则:**
- 编译层内的 `./` 相对路径保持 `./` 形式不变
- 编译层引用外部形参 (`A`, `B`, `C`) 替换为调用者传入的实参
- 立即数形参 (`1.0`, `true`) 替换为具体值
- 命名槽位 (`./mm`) 在 `/vthread/<vtid>/` 下平级创建，不受子栈嵌套影响

## 8. RETURN 处理 (translate.go 同文件)

```go
// handleReturn 处理 RETURN 指令，返回父栈 CALL 的下一条 PC
func handleReturn(ctx context.Context, rdb *redis.Client, vtid string, pc string) string {
    lastSlash := strings.LastIndex(pc, "/")
    if lastSlash < 0 {
        return pc // 根栈 return → vthread 即将 done
    }

    parentPC := pc[:lastSlash]

    // 1. 读取返回值写入父 CALL 的返回值槽位
    inst, _ := ir.Decode(ctx, rdb, vtid, pc)
    parentInst, _ := ir.Decode(ctx, rdb, vtid, parentPC)
    if len(parentInst.Writes) > 0 && len(inst.Reads) > 0 {
        retSlot := parentInst.Writes[0]
        retVal := inst.Reads[0]
        if strings.HasPrefix(retSlot, "./") {
            slotKey := "/vthread/" + vtid + "/" + retSlot[2:]
            rdb.Set(ctx, slotKey, retVal, 0)
        }
    }

    // 2. 递归 DELETE 当前子栈
    keys, _ := rdb.Keys(ctx, "/vthread/"+vtid+"/"+parentPC+"/*").Result()
    if len(keys) > 0 {
        rdb.Del(ctx, keys...)
    }

    // 3. PC 恢复到父栈 CALL 的下一条
    return ir.NextPC(parentPC)
}
```

```
RETURN 语义:
  输入: pc="[2,0]/[1,0]" → 在子栈 [2,0] 的第 1 条指令遇到 return

  1. parentPC = "[2,0]"
  2. 读取父 CALL 的 Writes[0]="./out" → 获得返回值 → 写入 /vthread/<vtid>/out
  3. KEYS /vthread/<vtid>/[2,0]/* → DEL key1 key2 ...
  4. 恢复 PC = "[3,0]" (父栈 CALL 的下一条)
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

## 10. 算子路由 (route/router.go)

```go
package route

import (
    "context"
    "encoding/json"
    "fmt"
    "math"
    "strings"

    "github.com/redis/go-redis/v9"
)

// Select 根据 opcode 选择负载最低的 op-plat 实例
func Select(ctx context.Context, rdb *redis.Client, opcode string) string {
    // 1. 找到支持该算子的程序
    programs, _ := rdb.Keys(ctx, "/op/*/list").Result()
    var chosenProgram string

    for _, progKey := range programs {
        list, err := rdb.LRange(ctx, progKey, 0, -1).Result()
        if err != nil {
            continue
        }
        for _, op := range list {
            if op == opcode {
                parts := strings.Split(progKey, "/")
                chosenProgram = parts[2] // "/op/op-cuda/list" → "op-cuda"
                break
            }
        }
        if chosenProgram != "" {
            break
        }
    }

    if chosenProgram == "" {
        panic(fmt.Sprintf("no op-plat supports opcode: %s", opcode))
    }

    // 2. 选择负载最低的实例
    instances, _ := rdb.Keys(ctx, "/sys/op-plat/*").Result()
    bestLoad := math.MaxFloat64
    bestInstance := ""

    for _, instKey := range instances {
        val, err := rdb.Get(ctx, instKey).Result()
        if err != nil {
            continue
        }
        var info struct {
            Program string  `json:"program"`
            Status  string  `json:"status"`
            Load    float64 `json:"load"`
        }
        json.Unmarshal([]byte(val), &info)

        if info.Status != "running" || !strings.Contains(instKey, chosenProgram) {
            continue
        }
        if info.Load < bestLoad {
            bestLoad = info.Load
            parts := strings.Split(instKey, "/")
            bestInstance = strings.TrimPrefix(parts[len(parts)-1], "op-")
        }
    }

    if bestInstance == "" {
        panic(fmt.Sprintf("no running op-plat instance for %s", opcode))
    }

    return bestInstance // e.g., "metal:0"
}
```

## 11. 子栈本地缓存 (engine/cache.go)

CALL 翻译完成后，将子栈 key-value 一次性 MGET 到本地 `map[string]string`。
生命周期与 Redis 子栈 key 一致：CALL 翻译后加载 → RETURN 时释放。

```go
package engine

import (
    "context"
    "fmt"
    "strings"

    "github.com/redis/go-redis/v9"
)

type SubStackCache struct {
    Root  string            // "/vthread/1/[2,0]/"
    Cells map[string]string // 相对 key → value, e.g., "[0,0]"→"matmul"
}

func LoadSubStack(ctx context.Context, rdb *redis.Client, vtid string, pc string) *SubStackCache {
    root := fmt.Sprintf("/vthread/%s/%s/", vtid, pc)

    keys, err := rdb.Keys(ctx, root+"*").Result()
    if err != nil || len(keys) == 0 {
        return nil
    }

    vals, err := rdb.MGet(ctx, keys...).Result()
    if err != nil {
        return nil
    }

    cache := &SubStackCache{
        Root:  root,
        Cells: make(map[string]string, len(keys)),
    }
    for i, key := range keys {
        localKey := strings.TrimPrefix(key, root)
        if s, ok := vals[i].(string); ok {
            cache.Cells[localKey] = s
        }
    }
    return cache
}

func (c *SubStackCache) Get(addr0, addr1 int) string {
    return c.Cells[fmt.Sprintf("[%d,%d]", addr0, addr1)]
}

func (c *SubStackCache) Release() {
    c.Cells = nil
}
```

**使用方式:** 进入子栈时加载缓存，后续指令解码走 `DecodeFromCache`，全程零 Redis GET。

## 12. 多 VM 并行

```
VM-0 拾取 /vthread/1 → 执行
VM-1 拾取 /vthread/2 → 执行
VM-2 空闲 → BLPOP notify:vm

原子抢占: WATCH/MULTI/EXEC 保证同一 vthread 只被一个 worker 拾取
负载均衡: 自然竞争，先到先得（与 op-plat 的 load-based 选择不同）
```

每个 VM 实例启动时注册:

```
SET /sys/vm/0 = {"status":"running", "pid":<pid>, "started_at":<ts>}
```

## 13. 性能优化

### 13.1 Pipeline 批写入

CALL 翻译时一次 Pipeline 写入整个子栈的所有 `[i,j]` key：

```
逐条 SET: N 次往返
Pipeline SET: 1 次往返
```

### 13.2 子栈本地缓存

```
逐条模式 (每次 GET):
  GET /vthread/1/[2,0]/[0,0]   ← Redis 往返
  → 20 条指令 ≈ 60 次 GET ≈ 6ms

子栈缓存 (一次 MGET):
  KEYS + MGET (1-2 次往返)
  → 后续 20 条指令: 0 次 GET ≈ 0ms
```

### 13.3 连接池调优

```go
rdb := redis.NewClient(&redis.Options{
    Addr:         "127.0.0.1:6379",
    PoolSize:     workers * 2,
    MinIdleConns: workers,
    PoolTimeout:  10 * time.Second,
    ReadTimeout:  3 * time.Second,
    WriteTimeout: 3 * time.Second,
})
```

### 13.4 批量发射 (TODO)

vthread 的 `mode` 字段为 `"batch"` 时，VM 向前扫描连续无依赖指令，一次 PUSH 多条到 op-plat。
依赖分析策略待定（编译器标注 vs VM 运行时分析）。

```go
// 预留
if state.Mode == "batch" {
    batch := collectBatch(rdb, vtid, pc)
    rdb.RPush(ctx, cmdQueue, marshalBatch(batch))
    // op-plat 消费整个 batch，完成后通知一次
}
```

## 14. 错误处理

```
op-plat 返回 error:
  1. BLPOP done:<vtid> → {"status":"error", "error":{...}}
  2. SET /vthread/<vtid> = {pc:pc, status:"error", error:{...}}
  3. Worker 释放该 vthread, 回到 idle

超时:
  BLPOP done:<vtid> 超时 (可配置, 默认 30s)
  → SET /vthread/<vtid> = {pc:pc, status:"error", error:{code:"TIMEOUT"}}

pysdk 可通过 GET /vthread/<vtid> 检查 status 字段感知错误。
```

## 15. 编译与运行

```bash
# Go 1.21+
go version

# 初始化项目
cd executor/vm
go mod init git.array2d.com/ai/deepx/executor/vm
go get github.com/redis/go-redis/v9

# 开发构建
go build -o ./bin/vm ./cmd/vm/

# 生产构建 (linux/amd64)
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o ./bin/vm ./cmd/vm/

# 运行 (需先启动 heapmem + op-plat)
./bin/vm

# 多 VM 实例
VM_ID=1 ./bin/vm
```

## 16. 验证方法

### 16.1 单元测试

```bash
go test ./internal/... -v
```

### 16.2 集成测试

```go
// +build integration

func TestIntegrationAdd(t *testing.T) {
    ctx := context.Background()
    rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:6379"})

    // 1. 准备数据
    rdb.Set(ctx, "/vthread/1/a", "1", 0)
    rdb.Set(ctx, "/vthread/1/b", "2", 0)

    // 2. 注册 func
    rdb.Set(ctx, "/src/func/add_test", "(add_test(A,B) -> (C))", 0)
    rdb.Set(ctx, "/src/func/add_test/0", "add(A, B) -> ./C", 0)

    // 3. 创建 vthread
    rdb.Set(ctx, "/vthread/2", `{"pc":"[0,0]","status":"init","mode":"single"}`, 0)
    rdb.Set(ctx, "/vthread/2/[0,0]", "call", 0)
    rdb.Set(ctx, "/vthread/2/[0,-1]", "add_test", 0)
    rdb.Set(ctx, "/vthread/2/[0,-2]", "./a", 0)
    rdb.Set(ctx, "/vthread/2/[0,-3]", "./b", 0)
    rdb.Set(ctx, "/vthread/2/[0,1]", "./c", 0)

    // 4. 唤醒 VM
    rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"2"}`)

    // 5. 轮询等待 done
    deadline := time.Now().Add(10 * time.Second)
    for time.Now().Before(deadline) {
        val, _ := rdb.Get(ctx, "/vthread/2").Result()
        var s struct{ Status string }
        json.Unmarshal([]byte(val), &s)
        if s.Status == "done" {
            c, _ := rdb.Get(ctx, "/vthread/2/c").Result()
            if c == "3" {
                t.Log("pass: 1+2=3")
                return
            }
            t.Errorf("expected 3, got %s", c)
            return
        }
        time.Sleep(100 * time.Millisecond)
    }
    t.Fatal("timeout")
}
```

## 17. 开发量评估

| 组件 | 新增代码 | 难度 | 说明 |
|------|---------|------|------|
| 项目骨架 + Redis 连通 + Worker Pool | ~100 行 | 低 | main.go + go.mod |
| Vthread 原子拾取 | ~60 行 | 中 | go-redis Watch + TxPipelined |
| 指令数据结构 + 解码 | ~120 行 | 中 | MGET 批量读取 + 本地缓存解码 |
| 指令分发 (核心) | ~180 行 | 高 | 状态机循环, goroutine 模型 |
| CALL eager 翻译 | ~200 行 | 高 | Pipeline 批量写子栈 |
| RETURN 处理 | ~100 行 | 中 | 子栈 DELETE + PC 恢复 |
| 算子路由 | ~80 行 | 中 | 负载感知实例选择 |
| 子栈本地缓存 | ~100 行 | 中 | MGET 加载 + map[string]string |
| Dxlang 解析器 | ~120 行 | 中 | 解析 `opcode(r1,r2) -> w1` |
| IF/FOR 控制流 | ~120 行 | 中 | 条件分支 |
| 错误处理 + 超时 | ~80 行 | 低 | context + 超时 |
| **合计** | **~1,260 行** | **高** | |

## 附录 A：与 C++ 方案的对比

| 维度 | C++ 版 | Go 版 |
|------|--------|-------|
| Redis 客户端 | hiredis (C, 手动封装) | go-redis/v9 (连接池/Pipeline/Tx 内置) |
| 并发模型 | 单线程 while(true) / pthread pool | goroutine worker pool (netpoller I/O 多路复用) |
| 阻塞 I/O 成本 | 1 个 pthread ≈ 1MB 栈 | 1 个 goroutine ≈ 2KB 栈 |
| 内存管理 | 手动 new/delete | GC 自动 |
| JSON | nlohmann/json 或手动封装 | encoding/json 标准库 |
| 构建 | CMake + hiredis 系统依赖 | `go build` 单二进制, 零外部依赖 |
| 测试 | Google Test / Catch2 | `go test` 标准库 |
| 跨平台 | 需配置交叉编译工具链 | `GOOS=linux GOARCH=arm64 go build` 一行 |
| 部署 | 动态链接 so/dylib | 静态链接单文件 |
| 代码量 | ~1,500 行 | ~1,260 行 |

## 附录 B：待确定问题

| 问题 | 状态 |
|------|------|
| 批执行依赖分析 (编译器 vs VM 运行时) | TBD，vthread mode 字段已预留 |
| 多 VM 负载均衡策略 | 自然竞争 (WATCH/MULTI/EXEC) |
| 子栈缓存失效 (外部修改) | 后续处理 |
| VM 数量与 Redis 连接池 | PoolSize = workers × 2 |
| 监控/可观测性 | 预留 `net/http/pprof`，不强制 |
| vthread 级 mode 字段 "single" \| "batch" | 已确认，py 侧创建时配置 |
