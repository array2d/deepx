## excuter

### op-mem-ompsimd

#### cpu的range算子辅助函数

range函数是shape类中的一个函数，用于根据shape对tensor进行omp线程并行遍历的方式

定义和实现分别在：

excuter/common/src/deepx/shape.hpp

excuter/common/src/deepx/shape_range.cpp

| func | omp并行 | omp线程local局部对象 | 调用场景   |
| ---- | ---- | ------ | ---------- |
|      | N    |        | print      |
| 函数 | 否   | 0      | 不需要并行 |
| 函数 | 是   | 0      | 需要并行   |
| 函数 | 否   | 0      | 不需要并行 |
