# deepx-go

 deepx-go是为了构建抽象计算图的golang接口库

供算法人员搭建模型使用

## 存算一体，控制分离

deepx-py是控制侧，负责生成计算图，通过通信调用其他进程

在executor中，实现了具体的存算引擎



## 接口设计
deepx-py的接口设计尽量贴合pytorch，方便用户进行迁移




