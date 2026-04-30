# deepx 速度赶超策略

> 分析 deepx/dxlang 如何在训练和推理速度上赶超 PyTorch + vLLM，覆盖单卡→多节点→大规模场景。

## 一、速度差距的根因分析

### 1.1 当前 deepx 的执行模型

```
dxlang 指令 → VM解释(每指令6次Redis往返) → op-plat(单算子dispatch) → GPU
```

**训练场景的具体差距**，以 Llama-70B 一次 forward+backward 为例：

| 维度 | PyTorch FSDP (8×H100) | deepx 当前架构 |
|------|----------------------|---------------|
| 每层 matmul 执行方式 | 直接 launch cuBLAS kernel | VM 6次Redis → RPUSH cmd → op-plat BLPOP → BLPOP done |
| 算子间数据传递 | GPU 显存直接传递 | 经过 Redis key + JSON marshal/unmarshal |
| 梯度同步 | NCCL all-reduce (900 GB/s NVLink) | 不存在 |
| 参数分片 | FSDP flatten + reduce-scatter | 不存在 |
| MFU (硬件利用率) | ~70% | <1% |
| 1T token 训练耗时 | ~3 天 | 理论不可完成（每op ~1ms Redis开销） |

**核心矛盾**：当前 deepx 每条 dxlang 指令都走 Redis 控制面，而 PyTorch 的训练图一旦编译完成，就完全在 GPU 上执行。Redis 的单次往返延迟 (~0.15ms) 对于 GPU 计算（一次 matmul 可能 <0.01ms 完成）是天文数字。

### 1.2 推理场景的差距

| 维度 | vLLM (8×H100) | deepx 当前架构 |
|------|--------------|---------------|
| 单请求延迟 | 预填充+解码 一体化 | VM 解释 + op dispatch，逐 token 串行 |
| KV-cache 管理 | PagedAttention (块级虚拟内存) | 不存在 |
| 批量调度 | Continuous batching (token级) | vthread 逐个执行 |
| 吞吐 (Llama-70B) | ~50K tok/s | <100 tok/s (粗略估) |

---

## 二、五个关键架构缺口（及填平方案）

### 缺口 1：从指令解释 → 图编译

**现状**：VM 逐条 `Execute()` 指令，每条 6 次 Redis 往返。

**方案**：在 dxlang 函数粒度做 AOT/JIT 编译。

```
当前:
  dxlang func → VM逐条解释 → 每指令 Redis dispatch → op-plat 单算子 → GPU

改造后:
  dxlang func → dxlang Compiler → Fused GPU Kernel → 一次 GPU launch
               ↑                              ↑
          Redis仅存源码                zero Redis in hot path
```

具体实现：

```
# 编译层新增 (缓存编译产物，避免重复编译)
/op/<backend>/kernel/<kernel_name>  = base64(compiled_kernel_binary)
/op/<backend>/kernel/<kernel_name>/meta = {"grid": [N,M], "block": [256], "smem": 48KB}

# VM CALL 时:
1. 检查 /op/cuda/kernel/<func_name> 是否存在
2. 存在 → 直接 launch kernel (一次 GPU call)
3. 不存在 → 编译: dxlang func → 融合 CUDA kernel → 存入 Redis → launch
```

**核心收益**：训练时每条指令从 ~1ms 降到 ~10μs（仅 GPU launch 开销），提升 **100×**。

### 缺口 2：无 GPU-to-GPU 通信

**现状**：GPU 间数据传输完全不存在。多卡 = 各算各的。

**方案**：新增 `comm-plat`（通信平面），封装 NCCL/RDMA。

```
op-plat 新增通信原语:
  allreduce(tensor_list, "sum") → reduced_tensors
  allgather(tensor) → gathered_tensors
  reduce_scatter(tensor) → scattered_tensors
  broadcast(tensor, src_rank) → replicated_tensor
  send(tensor, dst_node) / recv(src_node) → tensor
```

Redis 中的表示（仅存通信拓扑元信息，不存数据）：

```
/sys/topology/nodes     = ["n0:8×H100", "n1:8×H100", ...]
/sys/topology/links     = {"n0:n1": "IB_NDR400", ...}
/sys/topology/n0/gpus   = {"gpu0": "H100_80GB", ...}
```

数据本身通过 NVLink/InfiniBand 直通，Redis 只记录"谁和谁在通信"。

### 缺口 3：无分布式并行策略

**方案**：利用 dxlang 的声明式语义，在编译阶段自动插入并行策略。

```
# 用户在 dxlang 中写单卡逻辑:
def train_step(x:tensor, y:tensor) -> (loss:tensor):
    forward(x) -> ./h
    compute_loss(./h, y) -> ./loss
    backward(./loss) -> ./grad

# 编译器根据拓扑自动展开为 (以 2×4 TP+DP 为例):
# Node 0 GPU 0 (TP rank 0, DP rank 0):
def train_step_shard(x_shard_0:tensor, y:tensor) -> (loss:tensor):
    col_parallel_linear(x_shard_0, W_col_0) -> ./h0
    allreduce(./h0) -> ./h
    row_parallel_linear(./h, W_row_0) -> ./out0
    compute_loss(./out0, y) -> ./loss0
    backward(./loss0) -> ./grad0
    allreduce(./grad0) -> ./grad
```

Redis 核心价值：**所有参数在 KV 空间有全局路径**，编译器可以看到：

- 哪些 tensor 需要分片（shape 大 → 自动 TP）
- 哪些参数可以复制（小参数 → DP 广播）
- 数据怎么路由（`/data/shard_0` → node 0, `/data/shard_1` → node 1）

vs PyTorch：PyTorch 的 FSDP/DeepSpeed 配置分散在 rank 配置中，deepx 做到**配置集中 + 编译器自动决策**。

### 缺口 4：推理侧无 KV-cache 管理

**方案**：heap-plat 升级为 PagedAttention 式块管理器。

```
heap-plat 新增:
  alloc_block(block_size) → block_id      # 分配一个 KV-cache 块
  free_block(block_id)                     # 释放
  copy_block(src_id, dst_id)               # CoW 前缀共享
  defrag()                                 # 整理碎片
```

Redis 中的映射（轻量元信息）：

```
/heap/kv_cache/blocks/free          = [0, 1, 2, ...]        # List: 空闲块
/heap/kv_cache/blocks/used          = ["req_42:0→3", ...]   # 已分配映射
/heap/kv_cache/block_size           = 16                      # 每块 token 数
/heap/kv_cache/total_blocks         = 4096
```

vthread 天然映射到 continuous batching：

```
# VM scheduler 同时推进多个 vthread:
/vthread/req_1/   pc="[5,0]"  status="wait_decode"   # 等待采样
/vthread/req_2/   pc="[3,0]"  status="wait_decode"   # 同上
/vthread/req_3/   pc="[7,0]"  status="running"       # 正在执行 attention

# 一个 batch 中混合 prefill + decode，动态加入/退出
```

### 缺口 5：无混合精度 / 量化基础设施

**方案**：在 dxlang 类型系统层统一处理。

```
# 类型系统支持精度标注:
def forward(x:tensor<f16>, w:tensor<f16>) -> (y:tensor<f16>):
    matmul(x, w) -> ./y

# 编译器自动插入 cast (BF16 训练 / FP8 推理):
def forward(x:tensor<bf16>, w:tensor<bf16>) -> (y:tensor<bf16>):
    cast(x, master_fp32) -> ./x32        # 编译器插入
    cast(w, master_fp32) -> ./w32
    matmul(./x32, ./w32) -> ./y32
    cast(./y32, bf16) -> ./y
```

量化权重作为特殊 heap 变量，Redis 记录量化元信息：

```
/models/llama70b/layer0/q_weight   = {
    "dtype": "int4",
    "shape": [8192, 28672],
    "zero_point": 8,
    "scale": 0.037,
    "group_size": 128,
    "address": {"type": "shm", "shm_name": "/deepx_q_abc"}
}
```

---

## 三、多节点大规模：deepx 的结构性优势

PyTorch 的多节点方案本质是 **MPI 思想**：rank 编号、process group、显式通信。这在小规模（<100 GPU）效果好，但在以下场景吃力：

### 场景 A：异构集群（不同 GPU 型号混部 + CPU 节点）

PyTorch 的 FSDP 假设同构 GPU。deepx 的路径寻址天然支持异构：

```
# H100 节点上的 tensor 分片
/data/shard_h100_0  → node:0, device:gpu0 (H100, 80GB)
# A100 节点上的 tensor 分片
/data/shard_a100_0  → node:1, device:gpu0 (A100, 40GB)

# 编译器根据 device 能力自动调整分片大小:
# H100 → 分片更大，A100 → 分片更小
```

Redis 集中了"全局物理视图"，编译器做分片决策时无需 rank 协商。

### 场景 B：弹性训练（节点动态加入/退出）

```
# 节点加入:
/sys/topology/nodes  ← 追加 "n3:8×H100"
# 编译器重编译，redistribute shards
# 无需手动改 rank 配置

# 节点故障:
/sys/topology/nodes  ← 标记 n1 为 "dead"
# 自动触发 checkpoint 恢复 + 重分片
```

### 场景 C：超大规模 MoE（专家路由）

```
# MoE router 输出 (存在 Redis 中):
/vthread/req_42/route  = {"token_0": "expert_3", "token_1": "expert_7", ...}

# op-plat 消费路由信息，专家分布在不同节点:
/expert/3/forward  → node:5, gpu:2
/expert/7/forward  → node:2, gpu:0

# 通信模式: all-to-all (scatter tokens, gather results)
# Redis 只记录路由决策，token 数据走 RDMA
```

PyTorch 实现 MoE 需要手动管理专家映射和 all-to-all 通信。deepx 用路径空间做声明式路由。

---

## 四、量化对比：能否赶超？

### 训练速度

| 阶段 | PyTorch FSDP | deepx 改造后 | 关键 |
|------|-------------|-------------|------|
| 单卡单层 matmul | cuBLAS 直调 | 融合 CUDA kernel（同底层库） | 持平 |
| 多卡梯度同步 | NCCL all-reduce | comm-plat NCCL（同底层库） | 持平 |
| 图编译优化 | torch.compile (Inductor) | dxlang compiler（融合+内存规划） | 可达 90%+ |
| 异构调度 | 手动配置 | Redis 集中 + 自动分片 | **领先** |
| 弹性容错 | Elastic 启动器 | Redis 拓扑热更新 | **领先** |
| MFU 上限 | ~70% | ~65-70% | 接近持平 |

**结论**：训练速度，单卡硬上限相同（都是 cuBLAS/NCCL），**deepx 可以在 <16 节点内追平，>64 节点的异构/弹性场景可能反超**。

### 推理速度

| 阶段 | vLLM | deepx 改造后 | 关键 |
|------|------|-------------|------|
| Attention kernel | FlashAttention-3 | 复用（C++ FFI 对接） | 持平 |
| KV-cache 管理 | PagedAttention | heap-plat 块管理 | 可达同等 |
| Continuous batching | 内置调度器 | VM multi-vthread 调度 | 设计更灵活 |
| Prefix caching | 自动 | Redis 显式全局共享 | **领先** |
| 多模型混合服务 | 需多实例 | 共享 KV 空间路由 | **领先** |
| 吞吐上限 | ~50K tok/s | ~45-50K tok/s | 接近持平 |

---

## 五、路线图

```
Phase 1: 单卡图编译 (当前 → 3个月)
  ├─ dxlang func → fused Metal/CUDA kernel 编译器
  ├─ 保留 Redis 用于源码存储和 kernel cache
  └─ 目标: 单卡训练吞吐 ≥ PyTorch eager 的 80%

Phase 2: 单机多卡 (3个月 → 6个月)
  ├─ comm-plat: NCCL 封装为 opcode (allreduce / broadcast / reduce_scatter / allgather)
  ├─ heap-plat: FSDP-style shard/reconstruct (参数分片 + 收集)
  ├─ dxlang compiler: 自动插入 TP/DP 切分 pass
  └─ 目标: 8×H100 训练 MFU ≥ 60%

Phase 3: 多节点 (6个月 → 12个月)
  ├─ RDMA backend for comm-plat (InfiniBand / RoCE)
  ├─ 分层 all-reduce (node内 NVLink → node间 IB)
  ├─ Redis 拓扑管理 + 弹性容错 (热加入/热退出/自动恢复)
  └─ 目标: 64节点线性加速比 ≥ 85%

Phase 4: 推理优化 (并行 Phase 2-3)
  ├─ heap-plat PagedAttention (块分配/释放/CoW/碎片整理)
  ├─ VM continuous batching scheduler (多 vthread 并发推进)
  ├─ 量化路径 (AWQ/GPTQ → dxlang 类型系统 + 编译器自动 cast)
  └─ 目标: Llama-70B ≥ 40K tok/s
```

---

## 六、核心结论

**deepx 赶超 PyTorch/vLLM 的路径不是"做得更快"，而是"做得不同"**：

| 维度 | PyTorch/vLLM | deepx 改造后 |
|------|-------------|-------------|
| 编程模型 | 命令式 Python | 声明式 dxlang + 编译器 |
| 分布式协调 | 去中心化 (rank/group) | **集中式 KV 空间** + 去中心化执行 |
| 优化方式 | 手动配置 (FSDP/TP/PP) | **编译器自动决策** |
| 异构支持 | 勉强（同构假设，异构需手动调参） | **原生（路径寻址，按 device 能力自适应）** |
| 弹性 | 重启式（node failure → 全量重启） | **热更新（拓扑 KV 变更，增量恢复）** |
| 硬上限 | cuBLAS/NCCL 物理极限 | 同左（物理无法超越） |

**一句话**：在均质纯 GPU 集群上 deepx 只能追平，不可能超越（共享相同的数学和硬件上限）。但在 **异构混合集群、弹性训练、超大规模 MoE、多模型混合服务** 这些 PyTorch 架构不擅长的场景中，deepx 的集中式 KV 空间 + 声明式 dxlang 有结构性优势。

---

## 参考

- 当前 VM 吞吐分析: `.claude/skills/debug-kvspace.md`（每指令 6 次 Redis 往返，~800 native inst/s）
- Redis Key 布局: `doc/metaproc/redis-keys.md`
- dxlang 控制流设计: `doc/dxlang/spec-control-flow-v1.md`
- heap-plat 设计: `doc/heap-plat/README.md`
- op-plat 设计: `doc/op-plat/README.md`
