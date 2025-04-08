## 如何给excuter添加一个新算子

### 层次结构图

![层次结构图](./deepx.op.drawio.svg)


#### TensorFunction

顾名思义，TensorFunction是操作Tensor的函数，可以是c++函数，也可以是python函数，cuda函数等。

#### TensorFunction 特定精度特化，或混合精度实现


#### Op

Op是excuter的算子，是excuter的执行单元

在程序中，Op是基类，不同的Op有不同的实现，比如Add, Mul, MatMul等。
每个Op都需要override forward和backward函数

对同一个功能的Op如Matmul，可以有多种作者的实现

Matmul会选择选择一个默认的实现

或者由MatmulOp的name属性来指定具体author的实现
 

### 具体步骤

git clone https://github.com/deepx-org/deepx.git

#### 1.cpu执行器
cd deepx/excuter/op-mem-ompsimd

需要提前安装好依赖
+ highway需要源码安装
+ omp，openmp库
+ yaml-cpp
make build && cd build && cmake .. && make

你可以在test目录下，验证或添加测试用例


#### 2.cuda执行器
cd deepx/excuter/op-mem-cuda

需要提前安装好依赖
+ cuda
+ cublas
+ yaml-cpp

make build && cd build && cmake .. && make


#### 3.jax执行器

todo
 

#### 4.front对接测试

1.先启动excuter可执行文件， 位于excuter/op-mem-{cuda/ompsimd}/build，可执行文件名同excuter名
2.然后测试front中py的对应算子脚本（front/py/examples 目录）

可以按照顺序，以此测试

1_tensor

2_ir

3_functional



