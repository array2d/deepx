# exop-metal

> macOS Metal GPU 的 op-plat 实现。Apple Silicon 上使用 Metal Shading Language。
> 当前已有 1,325 行 C++/Metal 代码，需增加 Redis 命令循环和算子路由。

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| GPU | Apple Silicon (M1/M2/M3/M4) |
| 后端 | Metal 3 (MSL - Metal Shading Language) |
| 内存 | 统一内存 (CPU/GPU 共享) |
| 算子 | 已有: elementwise, activation, init |
| 缺 | matmul, reduce, changeshape |

## 2. 代码位置

```
executor/exop-metal/
├── CMakeLists.txt
├── src/
│   ├── client/main.cpp             入口 (Redis 消费者 + 计算指令分派)
│   ├── deepx/
│   │   ├── metal_context.hpp/cpp   Metal 设备管理 (MTLDevice, MTLCommandQueue)
│   │   ├── mem/mem_metal.hpp      内存 (shm → MTLBuffer 包装)
│   │   ├── dtype_metal.hpp        数据类型映射 (f32↔MTLDataTypeFloat)
│   │   └── tensorfunc/
│   │       ├── elementwise_miaobyte.hpp    add/sub/mul/div
│   │       ├── elementwise_common.hpp      relu/sigmoid/tanh/gelu
│   │       ├── init_miaobyte.hpp           zeros/ones/arange
│   │       ├── tensorlife_miaobyte.hpp     newtensor/deltensor/clone (历史代码)
│   │       └── metal_common.hpp            Metal 工具
│   └── test/shm/                 跨进程 shm 已验证

依赖 deepx-core:
  deepx_core 静态库 (dtype / tensor / shape / shmem / registry / stdutil)
  deepx_metal_hal (metal_device.hpp / metal_device.cpp, 查询 Metal 设备)
```

## 3. 算子清单

### 已实现

| 类别 | 算子 | GPU kernel 文件 |
|------|------|----------------|
| elementwise | add, sub, mul, div | elementwise_miaobyte.hpp |
| activation | relu | elementwise_common.hpp |
| activation | sigmoid | elementwise_common.hpp |
| activation | tanh | elementwise_common.hpp |
| activation | gelu | elementwise_common.hpp |
| init | zeros, ones, arange | init_miaobyte.hpp |

### 需开发

| 类别 | 算子 | 优先级 | 难度 |
|------|------|------|------|
| matmul | matmul | 高 | 中 — 推荐用 MPSMatrixMultiplication |
| reduce | sum, mean, max | 高 | 中 — Metal parallel reduction |
| changeshape | reshape | 中 | 低 — 零拷贝 (改 shape 元信息) |
| changeshape | transpose | 中 | 低 — Metal shader 索引重排 |
| changeshape | concat | 低 | 低 — memcpy 拼接 |
| changeshape | slice | 低 | 低 — 共享 shm + offset |
| activation | softmax | 中 | 中 — reduce + exp + div 组合 |
| norm | layernorm, rmsnorm | 低 | 中 |

## 4. 算子注册

启动时 (SET NX)：

```
/op/exop-metal/list = [
    "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu",
    "zeros", "ones", "arange",
    "matmul", "sum", "mean", "max",
    "reshape", "transpose", "concat", "slice"
]

/op/exop-metal/add = {
    "category": "elementwise",
    "dtype": ["f32", "f16", "i32"],
    "inputs": 2,
    "outputs": 1
}

/op/exop-metal/matmul = {
    "category": "matmul",
    "dtype": ["f32", "f16"],
    "max_shape": [8192, 8192, 8192],
    "fusion_group": "linear",
    "inputs": 2,
    "outputs": 1,
    "params": ["transpose_a", "transpose_b"]
}
```

## 5. Tensor 访问流程

```
1. 从指令获取 input key: "/vthread/1/a"
2. GET /vthread/1/a → tensor 元信息
3. shm_open(shm_name) + mmap → cpu_ptr
4. [device newBufferWithBytesNoCopy:cpu_ptr
        length:byte_size
        options:MTLResourceStorageModeShared
        deallocator:nil]
   → id<MTLBuffer> gpu_buf
5. GPU kernel 读写 gpu_buf
```

Apple Silicon 统一内存下，步骤 4 是零拷贝的。Metal 直接使用 CPU 物理内存页。

## 6. 算子调度 (dispatch_kernel)

```cpp
// 伪代码
json dispatch_kernel(string opcode, vector<TensorRef> inputs,
                     vector<TensorRef> outputs, json params) {

    if (opcode == "add") {
        auto a = inputs[0].mtl_buffer();
        auto b = inputs[1].mtl_buffer();
        auto c = outputs[0].mtl_buffer();
        auto N = inputs[0].element_count();

        auto encoder = command_buffer->computeCommandEncoder();
        encoder->setComputePipelineState(add_pipeline);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBuffer(c, 0, 2);
        encoder->dispatchThreads(MTL::Size(N, 1, 1),
                                  MTL::Size(256, 1, 1));
        encoder->endEncoding();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
    // ... 其他算子

    return {{"status", "ok"}};
}
```

## 7. matmul 实现方案

推荐使用 MPS (Metal Performance Shaders):

```objc
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// 初始化 (启动时)
MPSMatrixMultiplication* matmul_kernel = [[MPSMatrixMultiplication alloc]
    initWithDevice:device
    resultRows:M resultColumns:N interiorColumns:K
    alpha:1.0 beta:0.0];

// 每次调用
id<MTLBuffer> bufA = [device newBufferWithBytesNoCopy:a_ptr
    length:M*K*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
id<MTLBuffer> bufB = [device newBufferWithBytesNoCopy:b_ptr
    length:K*N*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
id<MTLBuffer> bufC = [device newBufferWithBytesNoCopy:c_ptr
    length:M*N*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];

MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
    columns:K rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
    columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
    columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

[matmul_kernel encodeToCommandBuffer:commandBuffer
    leftMatrix:matA rightMatrix:matB resultMatrix:matC];
```

## 8. 进程注册

```
/sys/op-plat/exop-metal:0 = {
    "program": "deepx-exop-metal-{hostname}-{pid}",
    "device": "gpu0",
    "status": "running",
    "load": 0.0,
    "pid": <pid>,
    "started_at": <ts>
}
```

## 9. 编译与运行

```bash
# 通过根 Makefile 统一构建
make build-exop-metal    # → /tmp/deepx/exop-metal/build/deepx-exop-metal

# 或手动 cmake
brew install hiredis
cd executor/exop-metal && mkdir build && cd build
cmake .. && make
```

## 10. 开发量

| 模块 | 新增代码 |
|------|---------|
| Redis 命令循环 | ~200 行 |
| 算子路由 dispatch | ~150 行 |
| Tensor 元信息映射 | ~100 行 |
| matmul (MPS) | ~150 行 |
| reduce kernel | ~100 行 |
| changeshape | ~50 行 |
| 算子注册 | ~50 行 |
| **合计** | **~800 行** |
