from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def reshape(t:Tensor,shape:list[int],out:Tensor,author='miaobyte')->Tensor:
    ir=DeepxIR("reshape", [t.node.name,shape], [out.node.name],author)
    send(ir)
    return out

def transpose(t:Tensor,dimorder:list[int],out:Tensor,author='miaobyte')->Tensor:
    ir=DeepxIR("transpose", [t.node.name,dimorder], [out.node.name],author)
    send(ir)
    return out
 
def concat(tensors:list[Tensor],dim:int,out:Tensor,author='miaobyte')->Tensor:
    ir=DeepxIR("concat", [[t.node.name for t in tensors], dim], [out.node.name],author)
    send(ir)
    return out 

def broadcastTo(t:Tensor,new_shape:tuple[int],out:Tensor,author='miaobyte')->Tensor:
    ir=DeepxIR("broadcastTo", [t.node.name,new_shape], [out.node.name],author)
    send(ir)
    return out
