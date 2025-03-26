from deepx.tensor import Tensor
from deepx.autograd.graph import Graph
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def newtensor(t:Tensor,name:str=None):
    graph = Graph.get_default()
    t._graph = graph
    t._node=graph.add_tensor(name,t=t)
    if t.graph.eager:
        ir2=DeepxIR("newtensor",[Param(t.shape)], [Param(t._node.name,category='tensor',precision=t.dtype)])
        send(ir2)
def copytensor(t:Tensor,out:Tensor):
    graph = Graph.get_default()
    out.node.add_input(t.node)
    if t.graph.eager:
        ir2=DeepxIR("copytensor", t.dtype, [t.node.name], [out.node.name])
        send(ir2)
def deltensor(t:Tensor):
    if t.graph.eager:
        ir2=DeepxIR("deltensor",'', [t.node.name], [])
        send(ir2)
