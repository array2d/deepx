from deepx.tensor import Tensor
from deepx.autograd.graph import Graph,default_graph
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def newtensor(t:Tensor,name:str=None):
    ir2=DeepxIR("newtensor",[Param(t.shape)], [Param(t._node.name,category='tensor',precision=t.dtype)])
    send(ir2)
    return t

def copytensor(t:Tensor,out:Tensor):
 
    ir2=DeepxIR("copytensor", t.dtype, [t.node.name], [out.node.name])
    send(ir2)
def deltensor(t:Tensor):
         
    ir2=DeepxIR("deltensor",'', [t.node.name], [])
    send(ir2)
