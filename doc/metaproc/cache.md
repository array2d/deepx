# 元程多级缓存一致性

> 传统 CPU 用 MESI 协议保证多核缓存一致。
> 元程的"缓存"是 executor 本地的路径缓存和 GPU 显存中的 tensor 副本。
> 同一个 tensor 可能被多个 executor 在不同机器上持有，需要分布式一致性协议。

## 1. 问题定义

```
元程 A 在 GPU0 (机器 n1) 写 heap:weights
元程 B 在 GPU1 (机器 n2) 读 heap:weights

问: 元程 B 读到的数据能否保证是最新版本？
```

这与 CPU 的缓存一致性问题完全相同，只是规模从几百个周期的总线嗅探变成了跨机器的网络消息。

## 2. 一致性模型

### 2.1 选择：因果一致性 + 显式版本控制

```
严格一致性 → 不可行 (跨机器延迟)
顺序一致性 → 性能太差
因果一致性 → ✅ 适合 ML 场景
  元程 A 写 weights → 通知元程 B → B 读 weights
  因果链保证 B 看到 A 的写入

最终一致性 + 版本号 → 灵活
  不阻塞写入，读者检查版本决定是否等待
```

### 2.2 版本号机制

每个 heap tensor 维护：

```
/vproc/<vpid>/heap/weights = {
    "shm":      "/deepx_t_xxx",
    "version":  42,           ← 单调递增版本号
    "owner":    "/vproc/1",   ← 当前写持有者
    "readers":  ["/vproc/2", "/vproc/3"],  ← 活跃读者
    "state":    "dirty"       ← dirty | clean | locked
}
```

## 3. 一致性协议：类 MESI 的分布式变体

```
传统 MESI 状态 (per cache line):
  Modified   — 本核独有, 已修改, 内存过期
  Exclusive  — 本核独有, 未修改, 与内存一致
  Shared     — 多核共享, 与内存一致
  Invalid    — 无效

元程 Tensor 状态 (per executor 本地):
  MODIFIED   — 本 executor 独写, version 最新, KVSpace 过期
  EXCLUSIVE  — 本 executor 独有, version 与 KVSpace 一致
  SHARED     — 多 executor 只读, version 与 KVSpace 一致
  INVALID    — 本地副本过期, 需重新获取
```

### 3.1 状态转换

```
  TENSOR_LOAD:
    INVALID ──GET Redis──→ SHARED (version=N)
    EXCLUSIVE ──────────→ SHARED (保持)

  TENSOR_STORE (写回):
    SHARED ──LOCK──→ MODIFIED (获取排他锁)
    写入 GPU buffer
    version++
    SET Redis {version: N+1, state:"dirty"}
    UNLOCK → EXCLUSIVE

  外部 WATCH 通知 (别的元程写了):
    SHARED ──version<N──→ INVALID (标记过期)
    下次 TENSOR_LOAD 时重新获取

  TENSOR_MOVE 到其他设备:
    EXCLUSIVE ──迁移──→ 目标设备 EXCLUSIVE
    源设备 → INVALID
```

### 3.2 协议消息

```
消息             发送方        接收方          含义
────             ──────       ──────          ────
INVALIDATE       写者         所有 SHARED 读者  我即将写，你们的副本作废
INVALIDATE_ACK   读者         写者             确认作废
READ_REQUEST     读者         写者/owner       我需要最新版本
READ_RESPONSE    写者         读者             这是最新数据 (含 version)
WRITEBACK        写者         KVSpace          更新 version，标记 clean
```

## 4. 多级缓存失效路径

```
元程 A 在 GPU0 执行 TENSOR_STORE heap:weights:

1. VM 调度器收到 STORE 指令
2. LOCK /tensor/weights/lock (Redis SETNX)
3. GET /vproc/A/heap/weights → {version:5, readers:[B,C]}

4. 发送 INVALIDATE 到读者 B 和 C:
   对每个读者:
     PUBLISH /vproc/<reader>/cache/invalidate {tensor:"weights", version:6}

5. executor B 收到 INVALIDATE:
   - 标记本地路径缓存中 weights 为 INVALID
   - 如果当前正在 GPU kernel 中读 weights → 等待完成
   
6. INVALIDATE_ACK 全部收到

7. GPU0 写入新数据
   SET /vproc/A/heap/weights.version = 6
   SET /tensor/weights/version = 6
   PUBLISH /tensor/weights {event:"updated", version:6}

8. UNLOCK /tensor/weights/lock
```

## 5. 路径缓存 (类似 TLB)

每个 executor 维护本地路径缓存：

```
┌─ Executor 路径缓存 ─────────────────────────────────────────┐
│                                                               │
│  KV路径              →  {本地GPU指针, byte_size, version, 状态} │
│  ────────                ─────────────────────────────────── │
│  /vproc/42/heap/w    →  {ptr:0x..., size:2MB, v:5, SHARED}  │
│  /vproc/42/stack/h   →  {ptr:0x..., size:4KB, v:1, EXCLUSIVE}│
│  /tensor/model/emb   →  {ptr:0x..., size:10MB, v:3, INVALID} │
│                                                               │
│  缓存策略: LRU                                                │
│  缓存大小: ~128 个条目                                        │
│                                                               │
│  查找: O(1) hash                                              │
│  命中率: >95% (tensor 访问局部性)                              │
│                                                               │
│  失效触发:                                                     │
│    - PUBLISH /vproc/+/cache/invalidate                        │
│    - 本地 version < Redis version                             │
│    - shm_unlink 检测                                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 6. 假共享与写合并

```
OS 假共享: 两个核写同一 cache line 的不同位置
元程假共享: 两个元程写同一 tensor 的不同切片

解决方案:
  - TENSOR_OP 支持 slice 操作数: heap:weights[0:512,:]
  - 不同切片独立锁: 无需等待整个 tensor
  - 切片级别的 version 号: /tensor/weights/version/0-512
```

## 7. 内存屏障

```
OS 内存屏障 (mfence): 保证屏障前的 store 对后续 load 可见
元程同步原语:

  BARRIER "step_done", count=N
    等待 N 个元程都到达屏障
    等价于: 全局同步点，之后所有元程的写入对彼此可见

  FLUSH heap:weights
    强制将本地 MODIFIED tensor 写回 KVSpace (version++)
    等价于: clflush / cache writeback

  INVALIDATE heap:weights
    强制标记本地副本为 INVALID，下次 TENSOR_LOAD 从 KVSpace 重新获取
    等价于: INVD / cache invalidate
```

## 8. 与传统缓存的精妙对应

```
CPU L1 Cache                元程 Path Cache
───────────                 ────────────────
Cache Line (64B)            Tensor Entry (任意大小)
Tag (地址高位)              KV Path (符号地址)
MESI State                  MODIFIED/EXCLUSIVE/SHARED/INVALID
Bus Snooping                Redis PUBLISH
Write-back                  FLUSH
Cache Miss → L2             未命中 → Redis → shm_open
Cache Coherence Protocol    Version-based Invalidation
LRU Eviction                LRU Eviction
False Sharing               Tensor Slice 独立锁
```

## 9. 为什么不做全局锁

传统分布式系统常用全局锁 (如 Redis Redlock)。但 ML 场景中 tensor 访问模式高度可预测：

```
forward  pass: 只读 weights，写 activations
backward pass: 写 gradients，读 activations
update:        写 weights (独占)

→ 编译期静态分析可知冲突模式
→ 运行时用 version + 轻量 invalidate，避免全局锁开销
→ 只在无法静态分析时回退到 LOCK/UNLOCK
```
