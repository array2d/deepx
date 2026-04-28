# 计算图


## 抽象计算图

抽象计算图是计算图的抽象表示，它描述了计算的整体逻辑结构。

## 执行计算图

执行计算图是计算图的实际执行过程，它描述了计算的详细具体执行过程。


自动tensor并行

+ 根据tensor的shape和dtype，对tensor进行split，分解为n个小tensor
+ 对每个小tensor，调度到不同的存算执行器上进行计算
+ 根据tensor的shape和dtype，对tensor进行concat





