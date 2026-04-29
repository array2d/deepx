# dxlang

dxlang是 deepx 体系中的“语言层基础库”。  
它负责定义深度学习tensor计算的统一的类型系统、结构化协议对象与可序列化语义，作为前端、编译器、调度器、执行器之间的公共语言。

和传统的单机进程完全不同，dxlang完全为分布式tensor运算而存在，其内存空间，代码，堆，栈，方法区的设计，通过redis存储管理索引+heap进程的显存存储tensordata，dxlang的线程/进程，天然就是运行在跨节点的底座中。

## 命名含义
- `dx`：领域前缀,deepx的缩写
- `lang`：语言层，强调语义与契约，而非具体硬件实现

## 定位
- 面向：前端 SDK、IR 工具链、执行器、存算面协议组件
- 提供：类型系统、张量元信息、协议对象、编解码与校验
- 不提供：硬件 kernel、显存/IPC 生命周期、调度策略

## 设计理念
1. 语言先于执行：先统一语义，再扩展执行器
2. IR-first：与 deepxIR 规范对齐，见 [docs/deepxIR/deepxir.md](docs/deepxIR/deepxir.md)
3. 稳定契约：跨进程、跨模块、跨后端保持一致语义
4. 执行解耦：不绑定 CUDA/Metal/ROCm 等实现细节
5. 最小依赖：保持轻量、可移植、可复用

### 2) 统一存算面地址空间
用于在统一寻址空间（如 Redis KV）与执行器之间传递的数据结构，例如：
- tensor 元信息记录（name/key、dtype、shape、device、bytes、ctime 等）
- 生命周期指令（create/get/delete 等）
从而便于调整优化控制流，适应不同的硬件集群环境

## 核心能力
- 类型系统：dtype、精度、类别与约束表达
- 结构模型：shape、tensor 元信息、协议载体
- 编解码：配置与对象的序列化/反序列化
- 校验：基础合法性检查与一致性约束

## 边界
- 不实现算子计算逻辑
- 不管理堆 tensor 分配/回收
- 不承担分布式调度与图优化

## 与其他组件关系
- front：生成图与语义对象，依赖 dxlang 契约
- compiler/scheduler：消费与变换语义，不破坏契约
- heap/op 执行器：实现运行时细节，遵循 dxlang 数据结构
- deepxIR：语义规范来源，见 [docs/deepxIR/deepxir.md](docs/deepxIR/deepxir.md)

## 目录
- `src/deepx/`：语言层核心定义
- `src/stdutil/`：通用工具
- `test/`：一致性与回归测试

## 构建
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```