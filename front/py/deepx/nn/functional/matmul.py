from typing import Optional,Union

from deepx import Tensor
from deepx.autograd import OpNode
from deepx.nn import DeepxIR
from deepx.scheduler import send
 

OpNode.register("matmul")

def matmul(
        a:Tensor,
        b: Tensor, 
        out:Union[Tensor,str]='',
        author:str='cublas'):
    opnode = a.graph.add_op("matmul")
    opnode.add_input(a.node)
    opnode.add_input(b.node)
    
    outtensor=None
    if isinstance(out,str):
        matmulshape=a.Shape.matmul(b.shape)
        outtensor=Tensor(shape=matmulshape, dtype=a.dtype, device=a.device)
        outtensor.addtograph(out)
    else:
        outtensor=out
    outtensor.node.add_input(opnode)
    if a.graph.eager:
        ir=DeepxIR("matmul", [a.node.name,b.node.name], [outtensor.node.name], author=author)
        send(ir)
    return outtensor
