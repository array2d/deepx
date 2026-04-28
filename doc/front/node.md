# Node,计算图的设计思考

## 概念

pytorch的计算图是动态的，tensorflow早期的计算图是静态的。

pytorch在前向传播时，会构建一个计算图，在反向传播时，会根据计算图进行反向传播。


## Graph结构

Node{
    froms []*Node
    tos []*Node
}

Graph结构可以支持Residual的跳跃Node连接

## Tree结构

Tree{
    parent *Node
    children []*Node
}

Tree结构需要特别的实现Residual的跳跃Node连接

Residual可以把跳跃连接的Node打平,都作为Residual的childs

## Deepx的设计实现 
优先考虑Tree这种静态图结构，如果需要支持Residual的跳跃Node连接，可以在forward和backward中特别的实现。