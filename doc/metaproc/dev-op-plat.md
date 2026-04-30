# op-plat 开发指南

> 开发 exop-metal (macOS Metal GPU)。op-cuda / op-cpu 暂不开发。
> heap-plat 完成后，有内存分配能力了再开发此组件。

## 1. 角色与职责

op-plat 是执行张量计算指令的被动进程。

| 能力 | 说明 |
|------|------|
| 指令消费 | 从 `cmd:exop-metal:0` 队列 RPOP/BLPOP 消费指令 |
| 张量计算 | 执行 GPU kernel (elementwise, matmul, reduce, changeshape) |
| 完成通知 | 计算完成后 LPUSH 到 `done:<vtid>` |
| 算子注册 | 启动时向 `/op/exop-metal/` 注册支持的算子 |

## 2. 当前状态

```
executor/exop-metal/
├── src/
│   ├── client/main.cpp             入口 (占位, 待改造)
│   ├── deepx/
│   │   ├── metal_context.hpp/mm   Metal 设备管理
│   │   ├── mem/mem_metal.hpp      内存管理 (shm 包装)
│   │   ├── dtype_metal.hpp        数据类型映射
│   │   └── tensorfunc/
│   │       ├── elementwise_miaobyte.hpp    add/sub/mul/div
│   │       ├── elementwise_common.hpp      relu/sigmoid/tanh/gelu
│   │       ├── init_miaobyte.hpp           zeros/ones/arange
│   │       ├── metal_common.hpp            Metal 工具函数
│   │       └── tensorlife_miaobyte.hpp     newtensor/deltensor/clone
│   └── test/shm/                  跨进程 shm 测试验证通过

代码量: 1,325 行
```

## 3. 通信模型

```
          VM                           exop-metal
          ──                           ────────
          PUSH cmd:exop-metal:0  ───→   RPOP/BLPOP 消费
          │                             │
          │                         根据 key 从 Redis GET tensor 元信息
          │                         通过 shm_name 获取 GPU 指针
          │                         执行 Metal GPU kernel
          │                             │
          BLPOP done:<vtid>  ←────── LPUSH 完成事件
```

## 4. 待开发任务

### 任务 O1: Redis 命令消费循环 (main.cpp)

```cpp
// 伪代码
while (true) {
    auto cmd = redis.rpop("cmd:exop-metal:0");
    if (!cmd) { usleep(100); continue; }

    auto req = json::parse(cmd);
    string opcode = req["opcode"];  // "add", "matmul", "relu", ...
    string vtid = req["vtid"];
    string pc = req["pc"];

    // 1. 获取 tensor 的 GPU 指针
    vector<void*> input_ptrs;
    for (auto& inp : req["inputs"]) {
        auto meta = redis.get(inp["key"]);          // GET tensor 元信息
        auto [fd, ptr] = shm_open_and_map(meta["address"]["shm_name"]);
        MTL::Buffer* buf = device->newBufferWithBytesNoCopy(ptr, meta["byte_size"], ...);
        input_ptrs.push_back(buf);
    }

    vector<void*> output_ptrs;
    for (auto& out : req["outputs"]) {
        auto meta = redis.get(out["key"]);
        auto [fd, ptr] = shm_open_and_map(meta["address"]["shm_name"]);
        MTL::Buffer* buf = device->newBufferWithBytesNoCopy(ptr, meta["byte_size"], ...);
        output_ptrs.push_back(buf);
    }

    // 2. 分发到 GPU kernel
    json result = dispatch_kernel(opcode, inputs, outputs, req["params"]);

    // 3. 更新输出 tensor 元信息 (如果 shape 变了)
    for (auto& out : req["outputs"]) {
        auto meta = redis.get(out["key"]);
        meta["version"] = meta["version"].get<int>() + 1;
        redis.set(out["key"], meta);
    }

    // 4. 通知 VM 完成
    redis.lpush("done:" + vtid, {
        {"pc", pc},
        {"status", result["status"]},  // "ok" or "error"
        {"outputs_updated", req["outputs"]}
    });
}
```

**依赖:** hiredis

### 任务 O2: Tensor 元信息获取

op-plat 根据 Redis key 获取 tensor 的 shm 地址:

```
指令中的 inputs: [{"key": "/vthread/1/a", ...}]
  → GET /vthread/1/a → {"dtype","shape","device","address":{"shm_name":"...","byte_size":...}}
  → shm_open(shm_name) → fd
  → mmap(fd, ..., byte_size) → CPU ptr
  → device->newBufferWithBytesNoCopy(ptr, byte_size, ...) → MTLBuffer (GPU ptr)
```

可选的本地路径缓存 (减少 Redis GET):
```cpp
struct PathCacheEntry {
    void* gpu_ptr;
    size_t byte_size;
    int version;
};
unordered_map<string, PathCacheEntry> path_cache;
```

当前版本每次重新 GET。缓存失效策略后续处理。

### 任务 O3: 完成通知

计算完成后，更新输出 tensor 元信息并通知 VM:

```cpp
// 更新输出 tensor version
redis.set("/vthread/1/c", updated_meta);

// 通知 VM
redis.lpush("done:" + vtid, json{
    {"pc", pc},
    {"status", "ok"},
    {"outputs_updated", json::array({"/vthread/1/c"})}
}.dump());
```

### 任务 O4: 算子注册 (程序级)

启动时向 `/op/exop-metal/` 注册支持的算子:

```
启动时注册:
  SET /op/exop-metal/list = [
      "add", "sub", "mul", "div",
      "relu", "sigmoid", "tanh", "gelu",
      "zeros", "ones", "arange"
  ]

  SET /op/exop-metal/add = {
      "category": "elementwise",
      "dtype": ["f32", "f16", "i32"]
  }

  SET /op/exop-metal/relu = {
      "category": "activation",
      "dtype": ["f32", "f16"]
  }

  SET /op/exop-metal/matmul = {
      "category": "matmul",
      "dtype": ["f32", "f16"],
      "max_shape": [8192, 8192, 8192]
  }

  注意: 算子注册在 /op/exop-metal/ (程序级)，与 /sys/op-plat/ (进程级) 分离
```

### 任务 O5: 算子路由 (dispatch_kernel)

```cpp
json dispatch_kernel(string opcode, vector<TensorRef> inputs,
                     vector<TensorRef> outputs, json params) {
    if (opcode == "add")      return kernel_add(inputs, outputs);
    if (opcode == "sub")      return kernel_sub(inputs, outputs);
    if (opcode == "mul")      return kernel_mul(inputs, outputs);
    if (opcode == "div")      return kernel_div(inputs, outputs);
    if (opcode == "relu")     return kernel_relu(inputs, outputs);
    if (opcode == "sigmoid")  return kernel_sigmoid(inputs, outputs);
    if (opcode == "tanh")     return kernel_tanh(inputs, outputs);
    if (opcode == "gelu")     return kernel_gelu(inputs, outputs);
    if (opcode == "zeros")    return kernel_zeros(inputs, outputs);
    if (opcode == "ones")     return kernel_ones(inputs, outputs);
    if (opcode == "arange")   return kernel_arange(inputs, outputs);
    if (opcode == "matmul")   return kernel_matmul(inputs, outputs, params);
    if (opcode == "sum")      return kernel_sum(inputs, outputs, params);

    return {{"status", "error"}, {"error", {{"code", "UNKNOWN_OP"}, {"message", opcode}}}};
}
```

### 任务 O6: 进程注册

启动时注册到 `/sys/op-plat/`:

```
SET /sys/op-plat/exop-metal:0 = {
    "program": "deepx-exop-metal-{hostname}-{pid}",
    "device": "gpu0",
    "status": "running",
    "load": 0.0,
    "pid": <getpid()>,
    "started_at": <timestamp>
}
```

## 5. GPU Kernel 覆盖情况

| 类别 | 已实现 | 需补充 |
|------|-------|--------|
| elementwise | add/sub/mul/div (miaobyte) | — |
| activation | relu/sigmoid/tanh/gelu (common) | — |
| init | zeros/ones/arange (miaobyte) | — |
| matmul | — | **需开发** (MPSMatrixMultiplication 或 Metal shader) |
| reduce | — | **需开发** (sum/mean/max) |
| changeshape | — | **需开发** (reshape/transpose/concat/slice) |

### matmul 开发建议

macOS 上矩阵乘法推荐使用 MPS (Metal Performance Shaders):

```objc
// 使用 MPSMatrixMultiplication
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
    initWithDevice:device
    transposeLeft:false
    transposeRight:true
    resultRows:M resultColumns:N interiorColumns:K
    alpha:1.0 beta:0.0];

[matmul encodeToCommandBuffer:commandBuffer
    leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
```

## 6. 编译与运行 (macOS)

```bash
# 安装依赖
brew install hiredis

# 构建
cd executor/exop-metal
mkdir -p build && cd build
cmake .. && make

# 运行
./deepx-exop-metal
```

## 7. 验证方法

```bash
# 终端1: 启动 exop-metal
./deepx-exop-metal

# 终端2: 通过 redis-cli 检查算子注册
redis-cli GET /op/exop-metal/list
# → ["add","sub","mul","div","relu","sigmoid","tanh","gelu","zeros","ones","arange"]

# 发送测试指令 (需先通过 heap-plat 创建 tensor)
redis-cli RPUSH cmd:exop-metal:0 '{
  "vtid":"test1",
  "pc":"[0,0]",
  "opcode":"add",
  "inputs":[
    {"key":"/test/a","dtype":"f32","shape":[100],"address":{"shm_name":"/deepx_t_xxx","byte_size":400}}
  ],
  "outputs":[
    {"key":"/test/c","dtype":"f32","shape":[100],"address":{"shm_name":"/deepx_t_yyy","byte_size":400}}
  ],
  "params":{}
}'

# 检查完成通知
redis-cli BLPOP done:test1 1
```

## 8. 开发量评估

| 任务 | 新增代码 | 难度 |
|------|---------|------|
| O1: Redis 命令循环 | ~200 行 | 中 |
| O2: Tensor 元信息获取 + shm 映射 | ~100 行 | 低 |
| O3: 完成通知 | ~50 行 | 低 |
| O4: 算子注册 | ~50 行 | 低 |
| O5: 算子路由 (dispatch) | ~100 行 | 低 |
| O6: 进程注册 | ~40 行 | 低 |
| matmul kernel | ~150 行 | 中 |
| reduce kernel | ~100 行 | 中 |
| changeshape kernel | ~50 行 | 低 |
| **合计** | **~840 行** | **中** |
