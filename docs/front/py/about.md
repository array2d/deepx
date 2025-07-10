### deepx/front/py

deepx-py库是DeepX框架的Python库，方便用户搭建深度学习模型，输出计算图，主要用于深度学习模型的开发和训练。

#### 设计理念

+ deepx并不像pytorch那样，追求python first，而是为了原生分布式和并行，约束python的灵活性。
+ deepx的使用风格，基本贴近pytorch。尽量能做到 import deepx as torch，依然能正确的run起来
+ deepx的py进程，不参与tensor计算，但会参与一些简单的shape计算

#### 待定

