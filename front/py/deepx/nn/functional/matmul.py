from typing import Optional,Union

from deepx import Tensor
from deepx.autograd import OpNode,Function,Context
from deepx.nn import DeepxIR
from deepx.scheduler import send
 

OpNode.register("matmul")
class Matmul(Function):
    @staticmethod
    def forward(ctx:Context,
                a:Tensor,
                b: Tensor, 
                out:Union[Tensor,str]='',
                author:str='cublas'):
        ctx.save_tensors(a,b)

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
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        a,b=ctx.get_tensors()
        return out_grad @ b.T, a.T @ out_grad

def matmul(a:Tensor,b:Tensor,out:Union[Tensor,str]='',author:str='cublas')->Tensor:
    return Matmul.apply(a,b,out,author)
