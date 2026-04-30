# op-plat 设计

> op-plat 是 DeepX 元程的 **计算平面**，负责执行张量运算。
> 本文档定义 op-plat 的抽象契约和所有实现的共用规范。

## 1. 定位

在元程 5 核架构中，op-plat 负责：

| 核心 | 角色 |
|------|------|
| VM | 将计算指令路由到 op-plat |
| op-plat | 被动消费指令，执行 GPU/CPU 张量计算 |
| Redis | 存储算子注册信息、tensor 元信息 |
| heap-plat | 提供 tensor 的物理存储 (shm) |

```
VM PUSH 计算指令 ──→ cmd:op-<backend>:<instance>
                        │
                   op-plat 消费
                        │
                   根据 key 从 Redis GET tensor 元信息
                   通过 shm_name 映射 GPU/CPU 指针
                   执行 GPU kernel
                   更新输出 tensor 元信息
                        │
VM BLPOP done:<vtid> ←── LPUSH 完成通知
```

## 2. 抽象契约

任何 op-plat 实现必须满足以下契约。

### 2.1 能力矩阵

| 能力 | 必须 | 说明 |
|------|------|------|
| 指令消费 | 是 | RPOP/BLPOP 从命令队列消费指令 |
| 张量计算 | 是 | 至少实现一种张量运算 |
| 完成通知 | 是 | LPUSH 完成事件到 done:<vtid> |
| Tensor 访问 | 是 | 根据元信息中 shm_name 映射物理地址 |
| 算子注册 | 是 | 启动时注册到 /op/<program>/list |
| 进程注册 | 是 | 启动时注册到 /sys/op-plat/<instance> |
| 批量执行 | 否 | 可选，支持 batch 指令并行执行 |

### 2.2 通信模型

```
消费: RPOP / BLPOP cmd:op-<backend>:<instance>
      ↓
解析: 提取 opcode, vtid, pc, inputs, outputs, params
      ↓
获取: GET <input_key> → tensor 元信息 → shm_open + mmap → GPU ptr
      ↓
执行: dispatch_kernel(opcode) → GPU kernel
      ↓
更新: SET <output_key> → 更新 tensor 元信息 (version++, shape)
      ↓
通知: LPUSH done:<vtid> {pc, status, outputs_updated}
```

### 2.3 指令格式

VM 发送到 `cmd:op-<backend>:<instance>` 的指令：

```json
{
    "vtid": "1",
    "pc": "[3,0]",
    "opcode": "matmul",
    "inputs": [
        {
            "key": "/vthread/1/a",
            "dtype": "f32",
            "shape": [1024, 512],
            "address": {
                "node": "n1",
                "device": "gpu0",
                "type": "shm",
                "shm_name": "/deepx_t_abc123",
                "byte_size": 2097152
            }
        }
    ],
    "outputs": [
        {
            "key": "/vthread/1/c",
            "dtype": "f32",
            "shape": [1024, 256],
            "address": {
                "node": "n1",
                "device": "gpu0",
                "type": "shm",
                "shm_name": "/deepx_t_def456",
                "byte_size": 1048576
            }
        }
    ],
    "params": {
        "transpose_a": false,
        "transpose_b": true
    }
}
```

**批量指令 (可选):**
```json
{
    "batch": [
        {"pc": "[3,0]", "opcode": "add", "inputs": [...], "outputs": [...]},
        {"pc": "[4,0]", "opcode": "mul", "inputs": [...], "outputs": [...]}
    ]
}
```

### 2.4 完成通知格式

```json
{
    "pc": "[3,0]",
    "status": "ok",
    "outputs_updated": [
        {"key": "/vthread/1/c", "new_shape": [1024, 256]}
    ]
}
```

错误:
```json
{
    "pc": "[3,0]",
    "status": "error",
    "error": {
        "code": "GPU_OOM",
        "message": "out of memory: requested 2GB, available 1.5GB"
    }
}
```

### 2.5 算子注册 (程序级)

op-plat **程序**的算子能力是静态的，所有**进程实例**共享：

```
/op/<program>/list          → JSON array    支持的算子列表
/op/<program>/<opname>      → JSON object   算子元数据
/op/<program>/func/<name>/  → dxlang text   编译产物 (编译器写入)
```

启动时注册——第一个实例负责写入 (SET NX)，后续实例只读取：

```
SET /op/exop-metal/list = ["add", "sub", "mul", "div", "relu", "sigmoid", ...]

SET /op/exop-metal/add = {
    "category": "elementwise",
    "dtype": ["f32", "f16", "i32"],
    "inputs": 2,
    "outputs": 1
}

SET /op/exop-metal/matmul = {
    "category": "matmul",
    "dtype": ["f32", "f16", "bf16"],
    "max_shape": [8192, 8192, 8192],
    "fusion_group": "linear",
    "inputs": 2,
    "outputs": 1,
    "params": ["transpose_a", "transpose_b"]
}
```

### 2.6 算子分类

| category | 算子示例 | 输入 | 输出 |
|----------|---------|------|------|
| elementwise | add, sub, mul, div | 2 | 1 |
| activation | relu, sigmoid, tanh, gelu | 1 | 1 |
| matmul | matmul | 2 | 1 |
| reduce | sum, mean, max | 1 | 1 |
| changeshape | reshape, transpose, concat, slice | N | 1 |
| init | zeros, ones, arange | 0 | 1 |
| fused | fused_matmul_add_relu | N | 1 |

### 2.7 进程注册

```
/sys/op-plat/exop-metal:0 = {
    "program": "deepx-exop-metal-{hostname}-{pid}",
    "device": "gpu0",
    "status": "running",
    "load": 0.3,
    "pid": <pid>,
    "started_at": <ts>
}
```

## 3. 消费者循环 (所有实现共用)

```
1. 启动:
   a) 设备初始化 (GPU context / Metal device)
   b) 算子注册: SET /op/<program>/list + /op/<program>/<opname>
   c) 进程注册: SET /sys/op-plat/<instance>

2. 循环:
   a) RPOP/BLPOP cmd:op-<backend>:<instance>
   b) 解析 JSON → opcode, vtid, pc, inputs, outputs, params
   c) 获取 tensor 指针:
      for each input:
        GET <key> → tensor 元信息
        shm_open(shm_name) + mmap → CPU ptr
        GPU context 包装 → GPU ptr (newBufferWithBytesNoCopy / cudaHostRegister)
   d) dispatch_kernel(opcode, inputs, outputs, params)
   e) 更新输出 tensor 元信息 (version++, shape)
   f) LPUSH done:<vtid> 完成通知

3. 退出:
   a) DELETE /sys/op-plat/<instance>
```

## 4. 各平台实现概览

| 实现 | 目录 | 状态 | GPU |
|------|------|------|-----|
| [exop-metal](exop-metal.md) | executor/exop-metal/ | 待改造 (已有 1,325 行) | Metal (Apple Silicon) |
| [op-cuda](op-cuda.md) | executor/op-cuda/ | 已成熟 (47 文件) | CUDA (NVIDIA) |
| [op-cpu](op-cpu.md) | executor/op-cpu/ | 待开发 | 纯 CPU |

## 5. 待确定问题

| 问题 | 状态 |
|------|------|
| 指令中发 GPU 指针 vs shm_name | VM 发完整 tensor 元信息，op-plat 自行映射 |
| 批量发射依赖分析 | 当前 VM 逐条发送 |
| op-plat 负载上报 | load 字段由 op-plat 自行更新到 /sys/ |
| 多 stream 并行 | Metal command queue / CUDA stream，待实现 |
| 路径缓存失效 | 当前每次重新 GET Redis |
