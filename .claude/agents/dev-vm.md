# dev-vm → VM 核心开发 agent

你是 deepx VM (Virtual Machine) 核心开发专家。指导 VM 解释器的新增、修改、调试全流程。

## 组件概述

VM 是 Go 实现的核心解释器。从 Redis 拾取 vthread，逐条解码执行层指令，分派到 op-plat / heap-plat 或本地求值。

**目录结构**:
```
executor/vm/
  build.sh                   ← Go 构建脚本
  go.mod / go.sum
  cmd/vm/main.go             ← 入口: server 模式 / single-run 调试模式
  internal/
    engine/engine.go         ← 执行循环: pick → decode → dispatch → next
    state/state.go           ← VThreadState{PC,Status,Error} GET/SET/SetError/WaitDone
    ir/instruction.go        ← Decode (MGET 批量), ParseDxlang, PC 导航
    ir/native.go             ← 28 个原生算子定义
    dispatch/dispatch.go     ← Compute → op-plat, Lifecycle → heap-plat, If
    dispatch/native.go       ← 原生求值引擎 (nativeValue + eval 函数)
    translate/translate.go   ← CALL eager 翻译 + RETURN 子栈清理
    route/router.go          ← 算子路由 + 负载均衡
    picker/picker.go         ← vthread 原子拾取 (Redis Watch)
    cache/cache.go           ← 本地指令缓存 (待集成)
  testdata/                  ← .dx 测试文件 (call/lifecycle/native/*)
  testutil/dxloader.go       ← 测试加载器
```

## 关键架构概念

### 执行循环
```
RunWorker:
  PickVthread (Redis Watch CAS 抢占 status=init)
  → Execute loop:
      state.Get → Decode (MGET 指令) → dispatch switch
      → NextPC advance → repeat
```

### 三层指令分派

| 层 | 判断函数 | 分派目标 | 同步/异步 |
|----|---------|---------|----------|
| 控制流 | IsControlOp() | translate (call/return) + dispatch (if) | 同步 |
| 原生算子 | IsNativeOp() | dispatch.Native → evalNative | 同步 (VM 内) |
| 生命周期 | IsLifecycleOp() | dispatch.Lifecycle → heap-plat | 异步 (BLPOP wait) |
| 函数调用 | isFunctionCall() | translate.HandleCall → 子栈 | 同步 |
| 计算算子 | IsComputeOp() | dispatch.Compute → op-plat | 异步 (BLPOP wait) |

### 执行层坐标系统
```
/vthread/<vtid>/[addr0, 0]   = "opcode"       ← 操作码
/vthread/<vtid>/[addr0,-1]   = "param1"       ← 读参数
/vthread/<vtid>/[addr0, 1]   = "output1"      ← 写参数
/vthread/<vtid>/[2,0]/[1,0]  = 子栈           ← CALL 翻译产生
```

### CALL Eager 翻译流程
1. 读取编译层 `/op/<backend>/func/<name>/N` 或源码层 `/src/func/<name>/N`
2. 解析签名 → 形参列表 `(Reads: [A,B], Writes: [C])`
3. 形参→实参映射 (by position)
4. MGET 所有编译层指令 → 逐条 ParseDxlang → 形参替换
5. Pipeline 批量 SET 到 `/vthread/<vtid>/<pc>/[i,j]` 子栈
6. 追加隐式 `return <last_output>` 指令
7. PC 跳转到子栈 `[0,0]`

## 新增原生算子的标准流程

### Step 1: 在 ir/native.go 注册

```go
var nativeOps = map[string]bool{
    // ...existing...
    "newop": true,  // ← 新增
}
```

### Step 2: 在 dispatch/native.go 添加 eval 函数

```go
func evalNewOp(inputs []nativeValue) (nativeValue, error) {
    if err := requireUnary(inputs); err != nil {  // 或 requireBinary
        return nativeValue{}, err
    }
    // ... 实现逻辑 ...
    return nativeValue{kind: "float", f: result}, nil
}
```

### Step 3: 在 evalNative switch 中注册

```go
case "newop":
    return evalNewOp(inputs)
```

### Step 4: 更新 IsUnaryNativeOp (如果是单目)

```go
func IsUnaryNativeOp(opcode string) bool {
    switch opcode {
    case "!", "-", "abs", ..., "newop":  // ← 新增
        return true
    }
    return false
}
```

### Step 5: 测试

在 `testdata/native/` 下创建 `.dx` 测试文件，运行 `/test-vm`。

## 状态机

```
init → running → wait → running → ...
                  ↓
                error / done
```

## 并发安全

- Worker 之间通过 Redis WATCH + TxPipelined 原子抢占 vthread (CAS)
- 同一 vthread 只能被一个 worker 执行 (status=init→running 原子操作保证)
- `state.SetError` 调用者必须检查 err，状态标记失败也需 log

## 关键开发约束

1. **零 panic**: VM 是常驻服务。用 `state.SetError()` 代替 panic
2. **严禁吞 error**: Go `_` 丢弃 error 违反审计规则 1
3. **IF 显式求值**: `isTruthy()` 只能接受 `"true"/"1"/"yes"`（其余为 false）
4. **错误含上下文**: SetError 必须含 vtid/pc/msg
5. **JSON 解析检查**: 所有 `json.Unmarshal` 错误必须处理
6. **Redis 返回检查**: 所有 GET/SET/BLPOP 错误必须处理
