# 元程设计对话记录

## 第一轮：元程概念抽象 (已归档，详见 git history)

将 OS 进程理论映射为元程：ISA、内存空间、缓存一致性、上下文、调用约定。
文档已删除，仅保留本轮的 spec-v1.md 和 deepx-design.md。

## 第二轮：元程架构重塑与 DeepX 实现设计 (2026-04-28)

### 核心决策

1. **KV 空间 = 元程边界**：一个 KV 空间实例就是一个元程（如同一个虚拟地址空间就是一个 OS 进程）。

2. **Vthread 寻址**：`/vthread/<vtid>/` 使用二维坐标 `[addr0, addr1]`：
   - addr0 = 栈序列索引 (指令位置)
   - addr1 = 0(操作码) / 负(读参数) / 正(写参数)
   - 命名槽位直挂 vthread 下：`/vthread/<vtid>/a = 1`

3. **CALL 语义**：复制 `/funcs/<name>/` 指令到 vthread 子栈，产生新维度。
   RETURN 时递归 DELETE 子栈。

4. **异步执行**：
   - VM PUSH → Redis LIST → op-plat RPOP 消费
   - op-plat LPUSH → done:<vtid> → VM BLPOP 唤醒
   - 四个核心进程全部被动执行，遵循 IR 管道机制

5. **堆的隐式命名空间**：除 `/vthread/`、`/funcs/`、`/sys/`、命令队列外，默认即堆。

6. **两个模式并存**：pysdk 直接发送 IR (当前) + 经编译器 (未来)。

### 当前文档

| 文件 | 内容 |
|------|------|
| `spec-v1.md` (505行) | 元程规范 v1 — 抽象模型 + DeepX 实现留白 |
| `deepx-design.md` (600行) | DeepX 实现设计 — Redis 约定、VM/op-plat/heap-plat 协议 |
| `CONVERSATION.md` | 本文件 |

### 待开发问题

- 编译器设计 (输入格式、依赖分析、算子融合)
- 多卡调度与负载均衡
- 指令格式优化 (物理地址 vs key 引用)
- 动态图支持
- 引用计数与 GC
- 错误恢复策略
