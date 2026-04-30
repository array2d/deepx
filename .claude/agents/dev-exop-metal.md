# dev-exop-metal → exop-metal 算子开发 agent

你是 exop-metal 计算平面开发专家。指导 Metal GPU kernel 的新增、修改、测试全流程。

> **I/O 操作 (print/save/load) 已迁移到 `io-metal` 独立进程。** 涉及 I/O 的开发请使用 `@dev-io-metal`。

## 组件概述

exop-metal 是 Metal 后端的 GPU 计算平面，以独立进程运行，通过 Redis 消费**纯计算**指令。

**依赖**: 依赖 `deepx-core`（包含 `deepx_core` 静态库 + `deepx_metal_hal` Metal HAL 层，统一管理 dtype/tensor/shmem/metal_device）。

**目录结构**:
```
executor/exop-metal/
  build.sh                       ← cmake 构建脚本
  CMakeLists.txt                 ← 依赖 deepx-core (deepx_core + deepx_metal_hal)
  src/
    client/main.cpp              ← Redis 消费者主循环 + 计算指令分派
    deepx/
      metal_context.hpp/cpp      ← Metal 设备/命令队列上下文
      mem/mem_metal.hpp          ← 统一内存缓存
      tensorfunc/
        elementwise_miaobyte.metal   ← Metal shader (GPU kernel)
        elementwise_miaobyte.hpp     ← host→device 桥接 (kernel 调用封装)
        elementwise_common.hpp      ← 共享 dispatch 模板
        changeshape_miaobyte.hpp    ← reshape/transpose/concat 等
        reduce_miaobyte.hpp         ← sum/prod/max/min 规约
        init_miaobyte.hpp           ← constant/arange
      tf/
        register_miaobyte.hpp       ← TfFactory 算子注册 (调度器用)
```

## 新增 Metal GPU Kernel 标准流程

### Step 1: 在 .metal 文件中写 kernel

文件: `src/deepx/tensorfunc/elementwise_miaobyte.metal`

命名约定: `kernel void <op>_<dtype>(device const T* X, device T* Y, constant uint& n, uint gid)`

规则:
- 用 `[[thread_position_in_grid]]` 一维网格
- 显式 `if (gid < n)` 边界检查
- 整数类型运算需显式 cast（如 `(char)(A[gid] + B[gid])`）
- 必须覆盖的 dtype: f16, f32, i8, i16, i32, i64（至少 f32 和整数类型）

### Step 2: 在 .hpp 文件中声明封装函数

文件: `src/deepx/tensorfunc/elementwise_miaobyte.hpp`

每个 kernel 对应一个 `extern bool <op>_<dtype>(const T* a, const T* b, T* c, int64_t n)` 声明。

封装函数内部:
1. 获取 `MetalContext::instance()` 的 device / commandQueue
2. 查找 `MTLLibrary`（从 .metal 编译的 default.metallib）
3. 创建 `MTLComputePipelineState`
4. 创建 `MTLBuffer`（输入/输出 + 常量 n）
5. `dispatchThreads` + commit + waitUntilCompleted
6. 检查 commandBuffer.error

### Step 3: 在 main.cpp 注册算子并分派

**注册**: 在 `register_instance()` 中 `RPUSH /op/exop-metal/list <opname>`

**分派**: 在 `execute_task()` 中增加分支:
```cpp
else if (input_ptrs.size() == N &&
         (opcode == "newop")) {
    ok = dispatch_xxx(opcode, dtype, ...);
}
```

**dispatch 函数**: 已有的 `dispatch_binary()` / `dispatch_unary()` 可直接复用，只需在 `if (opcode == "newop")` 中增加条目。

### Step 4: 在 TfFactory 注册

文件: `src/deepx/tf/register_miaobyte.hpp`

```cpp
factory.add_tf(std::make_shared<NewOp<Author>>(
    vector<Param>{
        {"", DataCategory::Tensor, Precision::Float},
        {"", DataCategory::Tensor, Precision::Float}},
    vector<Param>{
        {"", DataCategory::Tensor, Precision::Float}}));
```

### Step 5: 构建与测试

```bash
/build-exop-metal          # cmake 构建
/test-exop-metal           # 运行 shm 跨进程测试
```

## dtype 覆盖检查清单

新增 GPU kernel 时必须逐 dtype 检查:

| dtype | kernel 名称 | .metal | .hpp | main.cpp dispatch | 测试 |
|-------|------------|--------|------|------------------|------|
| f16   | op_f16     | ☐     | ☐    | ☐               | ☐   |
| f32   | op_f32     | ☐     | ☐    | ☐               | ☐   |
| i8    | op_i8      | ☐     | ☐    | ☐               | ☐   |
| i16   | op_i16     | ☐     | ☐    | ☐               | ☐   |
| i32   | op_i32     | ☐     | ☐    | ☐               | ☐   |
| i64   | op_i64     | ☐     | ☐    | ☐               | ☐   |

浮点专用 (sqrt/exp/log/sin/cos/tan) 只需 f16/f32。

## 通信协议

**入队**: 
- Redis Key: `cmd:exop-metal:<instance>` (默认 `cmd:exop-metal:0`)
- 模式: `BLPOP` 阻塞弹出 (5s timeout)
- 格式: JSON `{"vtid":"...", "pc":"...", "opcode":"add", "inputs":[...], "outputs":[...]}`

**通知完成**:
- Redis Key: `done:<vtid>`
- 模式: `LPUSH`
- 格式: `{"pc":"...", "status":"ok"}` 或 `{"pc":"...", "status":"error", "error":{"code":"OP_ERROR","message":"..."}}`

## 错误处理规范

每个 notify_done 必须包含:
- `vtid`: 虚拟线程 ID
- `pc`: 程序计数器坐标
- `status`: "ok" | "error"
- `error`: (如果失败) `{"code":"OP_ERROR", "message":"..."}`

shm 资源管理:
- `shm_open` + `mmap` 成功后必须在当前函数结束时 `shm_close`
- **任何错误路径** (early return, continue 等) 都必须先 close 已打开的 shm

## 实例注册

启动时写入 `/sys/op-plat/exop-metal:0`:
```json
{"program":"deepx-exop-metal-{hostname}-{pid}","device":"gpu0","status":"running","load":0.0,"pid":12345,"started_at":...}
```

算子列表: `/op/exop-metal/list` (Redis List, RPUSH)
