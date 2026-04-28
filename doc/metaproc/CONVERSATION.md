# 元程设计对话记录

## 会话摘要 (2026-04-28)

### 本轮对话的核心产出

**1. 架构关键决策**

- **KV 空间 = 元程边界**：一个 KV 空间实例就是一个元程（如同一个虚拟地址空间就是一个 OS 进程）。`/vthread/<vtid>/` 是元程内的线程。
- **没有 `/vproc/` 或 `/proc/`**：线程直接从 `/vthread/<vtid>/` 开始，入口为 `call main`。
- **没有 code/ 子树**：代码存于 `/funcs/` 下全局共享。vthread 栈帧按 CALL 递归创建和销毁。
- **堆的隐式命名空间**：除 `/vthread/`、`/funcs/`、`/sys/` 外，KV 空间的所有路径默认即堆。pysdk 自行划分命名空间。

**2. Vthread 寻址模型**

- **二维坐标 `[addr0, addr1]`**：addr0 = 栈序列索引，addr1 = 0(操作码)/负(读参数)/正(写参数)
- **命名槽位**：`/vthread/<vtid>/a`、`/vthread/<vtid>/b` 等，基础类型直存 Redis，tensor 存元信息
- **相对路径 `./`**：栈内局部变量使用 `./mm`、`./bias` 表示

**3. CALL/RETURN 语义**

- CALL 时**复制** `/funcs/<name>/` 的指令到 vthread 子栈，产生新维度：`/vthread/<vtid>/[n,0]/[0,0]`
- func 内部 `./` 相对路径保持，解析基点自动变为子栈
- 外部堆变量引用保持绝对路径
- RETURN 时**递归 DELETE** 子栈 KV 路径
- 嵌套调用形成多层维度：`/vthread/<vtid>/[0,0]/[3,0]/[0,0]`

**4. 异步执行模型**

- **VM → op-plat**：PUSH 到 op-plat 命令队列 (Redis LIST)
- **op-plat**：RPOP 消费指令，GPU 计算
- **op-plat → VM**：LPUSH 完成事件到 `done:<vtid>` 队列
- **VM**：BLPOP 等待完成通知
- 支持批量发射无依赖指令序列

**5. 四个核心进程**

VM 进程、op-plat、heap-plat、Redis。全部被动执行，遵循 IR 管道机制。

**6. `/funcs/` 设计**

- `/funcs/<name>` = 函数签名
- `/funcs/<name>/N` = 第 N 条指令
- 支持控制流：`/funcs/<name>/N/true/0`、`/funcs/<name>/N/false/0`
- 支持循环：`/funcs/<name>/N/body/0`

**7. 规范分层**

- **元程模型规范**：抽象语义，不依赖具体实现。定义 KV 空间要求、vthread 语义、op-plat/heap-plat 抽象契约
- **DeepX 实现规范**：Redis 具体路径约定、协议细节

### 关键设计对比

| 概念 | 之前设计 | 最终设计 |
|------|---------|---------|
| 命名 | `/vproc/<vpid>/` | `/vthread/<vtid>/` |
| 代码 | code/ 子树，存量程 | `/funcs/` 全局共享，CALL 时复制到栈 |
| 寻址 | 命名路径 (heap:name) | `[addr0, addr1]` 二维坐标 + `./` 相对路径 |
| 堆 | `/vproc/<vpid>/heap/` | 隐式：非保留路径即堆 |
| 调度 | VM 读 code → UDP 发送 | VM → Redis LIST → op-plat RPOP |
| 完成通知 | UDP 回复 | Redis LIST LPUSH → VM BLPOP |
| 元程定义 | deepx 专有概念 | 独立规范，deepx 是第一种实现 |

### 待开发问题

1. 编译器设计：pysdk → dxlang 编译 → 依赖分析 → TENSOR_FREE 插入 → 算子融合 → 数据并行
2. 多卡调度：多 op-plat 实例的负载均衡和设备选择
3. 指令格式优化：直接发送物理地址 vs 发送 key 让 op-plat 自己查询
4. 动态图支持：pysdk 可能发送动态图，编译器需运行时动态编译
5. 子栈复制开销：CALL 时 O(n) Redis 写操作是否需要优化
