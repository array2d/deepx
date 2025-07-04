### scheduler

DeepX框架的scheduler，是front和excuter之间的桥梁。

front只负责搭建抽象计算图，excuter负责执行算子，而scheduler负责将抽象计算图转换为执行计算图，并发送给excuter。

#### 算子注册器

算子注册器，接收excuter的算子及精度列表。


#### 调度器

scheduler将实现以下能力：

+ 根据计算图的依赖关系，确定算子的执行顺序。
+ 算子融合。抽象计算图都是由最基础的算子组成，而执行计算图可以由多个基础算子融合而成。
+ 算子消除。根据数学链式法则，有些算子可以相互抵消，如log和exp，mul和div,add和sub。
+ TP：tensor 并行，tensor自动拆分计算
+ PP：pipeline 并行,包括 dual-mode：前向和后向
+ MP：model 并行，模型自动拆分计算
+ DP：data 并行，多路batch并行训练


