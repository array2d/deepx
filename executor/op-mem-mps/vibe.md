# op-mem-mps 设计方案（Vibe）

> 目标：为 macOS / Apple Silicon 提供基于 Metal Performance Shaders (MPS) 的执行器，实现与现有执行器一致的接口与行为。

## 1. 目标与范围

### 1.1 目标
- 提供 MPS 后端执行器（op-mem-mps），对接 deepx 的 Tensor / TF / Op 体系。
- 兼容现有网络通信与任务调度流程（UDP server + TF factory）。
- 最小可用：支持设备探测、内存分配、张量生命周期与少量算子（init/io/elementwise）。

### 1.2 非目标
- 不包含跨平台抽象层的重构。
- 不覆盖全部算子，一期仅实现核心子集。

---

## 2. 总体架构

### 2.1 模块划分
- client：入口、网络服务、TF 注册与调度
- mem：MPS 设备/上下文与缓冲区管理
- tensorfunc：算子实现（以作者/精度为维度）
- tf：TF 封装与参数绑定

### 2.2 关键组件
1. MPSDevice
   - 枚举与选择 MTLDevice
   - 兼容 Apple Silicon 与 Intel + AMD（若支持）

2. MPSContext
   - 维护 MTLCommandQueue
   - 统一的 command buffer 生命周期

3. MPSBuffer / MemBase
   - 统一的 Tensor 内存管理
   - 与 deepx Tensor 对接

---

## 3. 数据流与执行流程

front(py) -> UDP -> TFFactory -> TF.run -> tensorfunc -> MPS -> output

- TF 负责参数解析与调度
- tensorfunc 负责具体算子
- mem 负责存储与同步

---

## 4. 目录与代码组织（建议）

executor/op-mem-mps/
  src/
    client/
      main.mm
      tfs.cpp
    deepx/
      mem/
      tf/
      tensorfunc/
      mps_device.{hpp,mm}
      mps_context.{hpp,mm}

---

## 5. API 设计与契约

### 5.1 与现有 executor 一致
- register_all(TfFactory&)
- TF::run(shared_ptr<MemBase>, string &error)
- MemBase::gettensor<T>()

### 5.2 MPS 约束
- 必须在 command buffer 提交后同步读回
- 统一使用 MTLStorageModeShared 以便 CPU 读取（一期）

---

## 6. 优先级路线图（MVP -> v1）

### MVP
- 设备探测 + MPSContext
- 张量生命周期 (new, delete)
- init (ones, zeros, arange)
- elementwise (add, mul)

### v1
- matmul
- reduce
- changeshape

---

## 7. 构建与依赖

- CMake + Objective-C++
- 依赖：
  - Metal / MetalPerformanceShaders
  - yaml-cpp（保持与其他执行器一致）

---

## 8. 风险与约束

- MPS 对部分精度支持有限（如 int8）
- Metal buffer 与 CPU 共享模式性能有限
- 异步执行与同步点控制复杂

---

## 9. 里程碑

| 阶段 | 内容 | 时间 |
|------|------|------|
| MVP | 设备+内存+基础算子 | 2 周 |
| v1  | 常用算子覆盖 | 4 周 |

---

## 10. 验证策略

- 对比 op-mem-ompsimd 输出
- 单元测试 + 前端 examples 回归

---

## 11. 需要确认的问题

1. 目标最小支持的算子集？
2. 是否允许 Metal shader 自定义 kernel？
3. MPSGraph 是否允许用于算子拼接？
