# 元程术语表 (Metaproc Glossary)

> 完整术语定义见 `doc/metaproc/spec-v1.md` 附录 C。
> 本文档为 Claude Code 上下文快速参考。

## 核心概念

| 术语 | 英文 | 定义 |
|------|------|------|
| 元程 | Metaproc | 一个 KV 空间实例，分布式计算的边界。类比 OS 进程。 |
| 元线程 | Vthread | 元程内的执行流。私有调用栈，共享堆数据。类比 OS 线程。 |
| KV 空间 | KV Space | Redis 实现的全局 key-value 存储。元程的运行载体。 |
| 路径空间 | Path Space | KV 空间的 key 命名规范。`/src/func/`、`/vthread/` 等。 |
| dxlang | DX Language | deepx 元程级编程语言（概念）；`executor/dxlang/` 是其当前 C++ 参考实现（代码）。 |

## 5 个核心 (Five Cores)

| 核心 | 英文 | 角色 |
|------|------|------|
| Redis | — | KV 空间：全局状态存储、命令队列 (List)、锁、通知 |
| pysdk | Python SDK | 算法前端：注册 dxlang 源码到 `/src/func/`，创建 vthread |
| op-plat | Operator Platform | 计算平面：被动消费指令，执行 GPU/CPU 张量运算 |
| heap-plat | Heap Platform | 堆管理平面：tensor 对象生命周期 (shm 创建/删除/克隆) |
| VM | Virtual Machine | 解释执行：CALL eager 翻译、指令路由到 op-plat/heap-plat、状态推进 |

## 三层 IR

| 层 | 路径 | 格式 | 角色 |
|----|------|------|------|
| 源码层 | `/src/func/<name>/` | dxlang 人类可读文本 | pysdk 写入 |
| 编译层 | `/op/<backend>/func/<name>/` | 编译器优化后 dxlang | 编译器写入，VM CALL 时读取 |
| 执行层 | `/vthread/<vtid>/` | `[addr0, addr1]` 二维坐标 | VM CALL 时 eager 翻译 |

## 路径空间

| 路径 | 说明 |
|------|------|
| `/src/func/<name>` | 函数签名 (dxlang) |
| `/src/func/<name>/N` | 第 N 条指令 |
| `/op/<backend>/func/<name>/N` | 编译后指令 (可能融合/拆分) |
| `/op/<program>/list` | 算子列表 (程序级, 所有实例共享) |
| `/op/<program>/<opname>` | 算子元数据 (category, dtype, max_shape...) |
| `/vthread/<vtid>` | vthread 自身: `{pc, status}` |
| `/vthread/<vtid>/[addr0,0]` | 操作码 (opcode) |
| `/vthread/<vtid>/[addr0,-N]` | 第 N 个读取参数 |
| `/vthread/<vtid>/[addr0,+N]` | 第 N 个写入参数 |
| `/vthread/<vtid>/<name>` | 命名槽位 (局部变量, 与指令坐标平级) |
| `/vthread/<vtid>/[n,0]/[0,0]` | 子栈 (CALL 产生) |
| `/sys/op-plat/<instance>` | op-plat 进程注册 |
| `/sys/heap-plat/<instance>` | heap-plat 进程注册 |
| `/sys/vtid_counter` | vthread ID 自增计数器 |
| `cmd:op-<backend>:<instance>` | op-plat 命令队列 (Redis List) |
| `cmd:heap-<backend>:<instance>` | heap-plat 命令队列 |
| `done:<vtid>` | vthread 完成通知队列 |
| `notify:vm` | VM 唤醒通知队列 |
| 其他非保留路径 | 堆变量 (tensor 元信息) |

## 指令格式

**dxlang 引号约定:**

| 引号 | 语义 | 示例 |
|------|------|------|
| `"..."` | 字符串字面量 | `"f32"`, `"[128]"`, `"hello"` |
| `'...'` | KV 空间 Key 路径 | `'/data/a'`, `'./ret'`, `'../parent'` |
| 无引号 | 变量名（形参/局部变量） | `A`, `alpha`, `my_var` |

**dxlang (源码层/编译层):**
```
opcode(read_p1, read_p2, ...) -> write_p1, write_p2
```

**执行层 (二维寻址):**
```
/vthread/<vtid>/[addr0, 0]  = "opcode"      ← addr1=0  → 操作码
/vthread/<vtid>/[addr0,-1]  = "param1"      ← addr1<0  → 读取参数
/vthread/<vtid>/[addr0, 1]  = "output1"     ← addr1>0  → 写入参数
```

`.` 相对路径 (`./mm`) 解析为 `/vthread/<vtid>/mm` (命名槽位)。

## Vthread 状态

| 状态 | 含义 |
|------|------|
| `init` | 已创建，待 VM 拾取 |
| `running` | VM 正在调度执行 |
| `wait` | 等待异步操作 (op-plat / heap-plat) |
| `error` | 执行出错 |
| `done` | 执行完毕，可 GC |

## CALL 语义

1. VM 读取 `/op/<backend>/func/<name>/` 编译层
2. 建立形参→实参映射
3. 逐条解析 dxlang → 形参替换 → 展开为 `[i,j]` 坐标
4. Pipeline 批量写入 `/vthread/<vtid>/[n,0]/` 子栈
5. PC 进入子栈首条指令

## RETURN 语义

1. 返回值写入父 CALL 指令的写参数槽位
2. 递归 DELETE 子栈 KV 路径
3. PC 恢复到父栈 CALL 的下一条

## 算子融合与拆分 (编译器)

| 操作 | 方向 | 说明 |
|------|------|------|
| 融合 (Fusion) | N→1 | 编译器将连续匹配指令替换为 fused 算子 |
| 拆分 (Split) | 1→N | Tensor 超过单卡上限时拆分 + 标注设备 |

两者都是编译器在 `/src/func/` → `/op/<backend>/func/` 层完成。

## Tensor 元信息

```json
{"dtype":"f32","shape":[1024,512],"byte_size":2097152,"device":"gpu0","address":{"type":"shm","shm_name":"/deepx_t_abc123"},"ctime":1714000000,"version":5}
```

## executor 目录

| 目录 | 角色 | 产出 |
|------|------|------|
| `deepx-core/` | 平台无关公共库：dtype / tensor / shmem / registry / stdutil / tensorfunc / tf / mem | `libdeepx_core.a` |
| `common-metal/` | **Metal HAL**：检测 macOS GPU 是否支持 Metal（`metal_device`），供 exop-metal / heap-metal 使用 | `libdeepx_metal_hal.a` |
| `exop-metal/` | Metal GPU 算子引擎（op-plat 实现之一） | `deepx-exop-metal` |
| `exop-cpu/` | CPU 算子引擎（op-plat 实现之一，无 HAL 依赖） | `deepx-exop-cpu` |
| `heap-metal/` | Metal GPU 堆管理（heap-plat 实现之一） | `deepx-heap-metal` |
| `heap-cpu/` | CPU 堆管理（heap-plat 实现之一，无 HAL 依赖） | `deepx-heap-cpu` |
| `io-metal/` | tensor I/O 平面（print/save/load） | `deepx-io-metal` |
| `vm/` | VM 解释执行：CALL 翻译、指令路由 | `deepx-vm` |

## 与 OS 进程对照

| OS 概念 | 元程对应 |
|---------|---------|
| 虚拟地址空间 | KV 空间 |
| 进程 | 一个 KV 空间实例 |
| 线程 | Vthread |
| 代码段 (.text) | /src/func/ + /op/<backend>/func/ |
| 堆段 (.data/.bss) | 非保留路径 (堆变量) |
| 栈段 | /vthread/<vtid>/ |
| PC | /vthread/<vtid> 的 pc 字段 |
| CALL/RET | CALL 翻译 → 子栈 / RETURN → DELETE 子栈 |
| 系统调用 | heap-plat / op-plat 命令 |
