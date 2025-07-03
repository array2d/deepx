## c++:计算执行器(excuter)

负责实现tensor的具体计算过程,对接硬件如GPU、CPU的simd指令

除了c++，也就只有编译器能干这样的脏活累活了

deepx用到了以下库，都是c++是实现

cblas
openmp
c++可以和汇编结合，从而最大程度发挥cpu、gpu寄存器的性能

cuda是c++的语言子集，也可以看作是c++


## python:模型前端构建
python提供了类似pytorch的库，便于调试和验证模型算法

deepx/tensor/
deepx/nn/deepxIR
deepx/nn.module/
deepx/nn.functional
通过这些库，我们可以快速的搭建一个模型结构

## golang:运维、监控、分布式，深度学习训推自动化的维护者

与pytorch、tensorflow不同，deepx追求分布式过程自动化，因此python侧不参与分布式

deepxctl:提供对deepx体系的所有工具、库、模型、镜像的统一纳管



## deepxIR
虽然deepxIR不是独立的编程语言，但是deepx体系的程序格式标准

excuter所执行的内容，就是deepxir的序列或deepxir计算图

https://github.com/array2d/deepx/blob/main/doc/excuter/op-mem-cuda/list.md

deepxir分为4类

计算：tensor这些系列elementwise、changeshape、tensorlife、io、reduce、init

指令结构:
queue[deepxIR]，串行指令，有前后执行顺序
parallel[deepxIR]，可并行的指令，无顺序依赖，可并行
以上指令为静态图所需的指令，运行过程是确定的。

分支：goto、ifelse
分支指令会让计算图行为不可预测，也就是动态部分

控制：parse、run等特殊自定义指令
控制指令是deepx分布式系统内置的各个组件控制指令