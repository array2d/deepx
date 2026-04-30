# VM 设计

> VM 是 DeepX 元程的 **解释执行引擎**，负责调度 vthread、翻译 CALL、路由指令。
> VM 是 5 核架构中的核心调度者，连接 pysdk、op-plat、heap-plat。

**实现语言：Go**

## 1. 定位

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
| heap-plat | VM PUSH 生命周期指令到 cmd:heap-* |

## 2. 项目结构

```
executor/vm/
├── go.mod / go.sum
├── cmd/vm/main.go                 入口: Redis 连接, 注册 VM, 启动 worker pool
├── internal/
│   ├── engine/engine.go           核心编排: RunWorker, Execute, dispatchControl
│   ├── state/state.go             VThreadState + Redis 状态读写
│   ├── picker/picker.go           Vthread 原子拾取 (WATCH/MULTI/EXEC)
│   ├── dispatch/
│   │   ├── dispatch.go            指令分发 (compute / lifecycle / if)
│   │   └── native.go              原生算子求值 (算术/比较/逻辑)
│   ├── translate/translate.go     CALL eager 翻译 + RETURN 处理 + 签名解析
│   ├── ir/
│   │   ├── instruction.go         指令数据结构 + 解码 + ParseDxlang
│   │   └── native.go              原生算子注册表
│   ├── route/router.go            算子路由 (负载感知选实例)
│   └── cache/cache.go             子栈本地缓存 (可选优化)
├── testdata/*.dx                  dxlang 测试函数
├── testutil/                      测试工具 (LoadDxFile, RegisterFunc, CreateVThread)
├── engine_test.go                 单元测试
└── integration_test.go            集成测试
```

依赖: `github.com/redis/go-redis/v9`

包依赖关系 (无循环):
```
engine → picker, state, dispatch, translate, ir, route
picker → state
dispatch → state, ir, route
translate → ir, route
state → (rdb only)
ir → (rdb only)
route → (rdb only)
```

## 3. 并发模型

单进程内启动 N 个 worker goroutine (默认 `GOMAXPROCS`)。
每个 worker 独立循环：拾取 vthread → 执行到底 → 再拾取。
多 worker 通过 Redis WATCH/MULTI/EXEC 自然竞争，无需中心调度器。

### 状态机

```
         start → 注册 /sys/vm/<id>
           │
    ┌──────▼──────┐
    │    idle     │◄──────────────────┐
    │ PickVthread │                   │
    │ BLPOP wait  │                   │
    └──────┬──────┘                   │
           │ status=init              │
    ┌──────▼──────┐                   │
    │   running   │                   │
    │   Execute   │                   │
    └──────┬──────┘                   │
           │                         │
    ┌──────┴──────┐                  │
    │             │                  │
 计算指令    控制流/原生              │
    │             │                  │
  PUSH →    VM 直接处理              │
  op-plat   (call/if/return)        │
    │         + native eval          │
  BLPOP         │                   │
  done          │                   │
    │             │                  │
    ▼             ▼                  │
  PC++          PC++                │
    │             │                  │
    └──────┬──────┘                  │
           │ vthread done           │
           └─────────────────────────┘
```

## 4. 核心执行循环 (engine.go)

```go
func Execute(ctx, rdb, vtid) {
    for {
        s := state.Get(vtid)
        if s.Status == "done" || "error" { return }

        inst := ir.Decode(vtid, pc)
        switch {
        case IsControlOp:   dispatchControl(...)    // call / return / if
        case IsNativeOp:    dispatch.Native(...)    // + - * / == < && !
        case IsLifecycleOp: dispatch.Lifecycle(...) // newtensor / deltensor
        case IsFunctionCall: → convert to "call" internally
        case IsComputeOp:   dispatch.Compute(...)   // → op-plat
        }
    }
}
```

### 函数调用检测

dx 源代码中调用另一函数无需 `call` 关键字，直接使用函数名：

```dxlang
// caller.dx — 直接调用 callee
def caller(A:int, B:int) -> (C:int) {
    callee(A, B) -> ./C     // ✅ 无需 call()
}
```

引擎在 Execute 中自动检测：若 opcode 不是内置关键字（native/control/lifecycle），
则检查 `/src/func/<name>` 或 `/op/*/func/<name>` 是否存在 → 存在则视为函数调用，
内部转换为 `call(funcName, args...)` 格式执行。

## 5. Vthread 拾取 (picker.go)

多 worker 并行竞争，Redis 事务保证每条 vthread 只被一个 worker 拾取：

```go
func tryPick(vtid) bool {
    rdb.Watch(func(tx) {
        state := tx.Get(key)
        if state.Status != "init" { return errSkip }
        state.Status = "running"
        tx.Set(key, state)
    })
    return err == nil
}
```

- 无可用 vthread: BLPOP notify:vm (5s 超时)
- `errSkip` 哨兵: 非 init 状态 vthread 跳过

## 6. 指令解码 (ir/instruction.go)

执行层坐标格式: `[addr0, addr1]`，addr1 负数为 reads，正数为 writes。

```
/vthread/<vtid>/[3, 0] = "add"       // opcode
/vthread/<vtid>/[3,-1] = "./a"       // read[0]
/vthread/<vtid>/[3,-2] = "./b"       // read[1]
/vthread/<vtid>/[3, 1] = "./c"       // write[0]
```

子栈 PC 格式: `[parent_addr0,0]/[child_addr0,0]`

**ParseDxlang** 支持三种格式:
- 前缀: `add(A, B) -> ./C`
- 中缀: `A + B -> ./C`
- C风格: `./C <- A + B`

```go
type Instruction struct {
    Opcode string   // "+" | "call" | "return" | "matmul" | funcName ...
    Reads  []string
    Writes []string
    PC     string
}
```

## 7. CALL Eager 翻译 (translate.go)

CALL 时一次性将编译层 dxlang 翻译为执行层 `[i,j]` 坐标。后续逐条执行零解析开销。

### 翻译流程

```
handleCall(funcName="add_test", args=["./a","./b"], out=["./c"]):
  1. 确定 backend (exop-metal > op-cuda > op-cpu)
  2. 读取签名: GET /src/func/add_test → "def add_test(A:int, B:int) -> (C:int)"
  3. 解析形参: reads=["A","B"], writes=["C"]
  4. 建立绑定: {A:"./a", B:"./b", C:"./c"}
  5. MGET 函数体 (按数字后缀排序, 保证指令顺序):
       /src/func/add_test/0 = "add(A, B) -> ./C"
  6. 逐条翻译 → Pipeline SET:
       /vthread/1/[0,0]/[0,0] = "add"
       /vthread/1/[0,0]/[0,-1] = "./a"  (A → 实参替换)
       /vthread/1/[0,0]/[0,-2] = "./b"
       /vthread/1/[0,0]/[0,1] = "./C"   (输出槽位, ./ 保持)
  7. 追加隐式 return: return(./C)
  8. Pipeline Exec (1 次 Redis 往返)
```

### 隐式 RETURN

每个函数体末尾自动追加 `return(./<outputParam>)` 指令，将最后一个输出形参的值回传给父栈。

### mgetAll 排序

`KEYS` 返回顺序不确定，使用 `strconv.Atoi` 按数字后缀排序后再 MGET。

## 8. RETURN 处理

```
handleReturn(pc="[0,0]/[1,0]"):
  1. parentPC = "[0,0]"
  2. 当前指令 reads[0]="./C" → 解析为实际值:
       GET /vthread/<vtid>/C → "15"
  3. 父 CALL 指令 writes[0]="./c" → 写入:
       SET /vthread/<vtid>/c = "15"
  4. 删除子栈: KEYS /vthread/<vtid>/[0,0]/* → DEL ...
  5. 恢复 PC = NextPC("[0,0]") = "[1,0]"
```

## 9. 参数约定

### 参数 key 命名空间

```
栈内变量 (vthread-local):
  ./a, ./b, ./tmp         相对路径 → /vthread/<vtid>/a
  A, B, X                 普通变量名 → 由 replaceParams 替换为实参

堆对象 (全局):
  /models/W, /heap/...    绝对路径 → 全局堆地址
  /data/weights.bin       跨 vthread 共享
```

**规则:**
- `./` 开头的 key 是栈内变量，解析时补全为 `/vthread/<vtid>/<name>`
- 普通变量名 (无 `./` 前缀) 是形参占位符，翻译时替换为调用者传入的实参
- `/` 开头的 key 是堆对象全局路径，不做栈内补全

### 三类参数解析

| 参数格式 | 解析方式 | 示例 |
|---------|---------|------|
| `./name` 相对路径 | → `/vthread/<vtid>/name` | ./mm → /vthread/1/mm |
| 普通变量名 | → replaceParams 替换为实参 | A → ./a |
| `/heap/...` 绝对路径 | 直接使用 | /models/W |
| 立即数 | 直接使用 | `1.0`, `true` |

## 10. 原生算子求值 (dispatch/native.go)

18 个符号算子直接在 VM 内求值，不经过 op-plat:

| 类别 | 算子 |
|------|------|
| 算术 | `+` `-` `*` `/` `%` |
| 比较 | `==` `!=` `<` `>` `<=` `>=` |
| 逻辑 | `&&` `\|\|` `!` |
| 位运算 | `&` `\|` `^` `<<` `>>` |

类型感知求值: bool → int → float → string，自动提升。
- 算术: 两边 int → int 结果；否则 → float
- 除法: 始终 float
- 比较: 数值优先，回退字符串

## 11. 算子路由 (route/router.go)

```go
Select(opcode) → "metal:0":
  1. 扫描 /op/*/list, 找到支持该 opcode 的程序
  2. 扫描 /sys/op-plat/*, 选该程序下负载最低的实例
  3. 返回 instance 标识符, e.g., "metal:0"
```

## 12. 编译器无关的函数调用

dx 源文件 (.dx) 以 `def` 关键字定义函数:

```dxlang
# callee.dx — 被调用函数
def callee(X:int, Y:int) -> (Z:int) {
    X + Y -> ./Z
}

# caller.dx — 直接调用 callee，无需 call 关键字
def caller(A:int, B:int) -> (C:int) {
    callee(A, B) -> ./C
}
```

引擎在运行时自动识别 `callee` 为已注册函数，将其转换为 CALL 指令。

## 13. 错误处理

```
op-plat 返回 error:
  → SET /vthread/<vtid> = {status:"error", error:{...}}

超时:
  BLPOP done:<vtid> 超时 (30s)
  → SET /vthread/<vtid> = {status:"error", error:{code:"TIMEOUT"}}

解码/执行异常:
  → setError() 标记 error 状态
```

## 14. 编译与运行

```bash
cd executor/vm
go build -o ./bin/vm ./cmd/vm/
./bin/vm
# 多实例: VM_ID=1 ./bin/vm
```

## 15. 验证

```bash
# 单元测试
go test ./... -v

# 集成测试 (需 Redis + mock op-plat)
go test -tags=integration -v -run 'TestIntegration'
```
