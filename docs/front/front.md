
# deepx前端

## 对应关系

| 前端 | pytorch | tensorflow | deepx|
| --- | --- | --- | --- |
| tensor库 | ATen | TensorFlow | deepx/tensorfunc | 
| 算子(支持forward和backward) | torch.nn.functional | ？ | deepx/op |
| 计算图子图| torch.nn.Module | tensorflow.nn.Module | deepx.nn.Module | 
| 抽象计算图 | torch.fx.graph.Graph | ？| deepx.nn.Graph | 
| 执行计算图| torch._inductor.graph.GraphLowering | tensorflow.Graph | deepx.nn.Graph | 