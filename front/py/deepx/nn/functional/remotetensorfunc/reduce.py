from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def sum(a:Tensor,dim:tuple[int],keepdim:bool,out: Tensor, author:str='miaobyte')->Tensor:
    ir2=DeepxIR("sum",[a.node.name,dim,keepdim], [out.node.name], author)
    send(ir2)
    return out
    
def prod(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    ir2=DeepxIR("prod",[a.node.name,dim,keepdim], [out.node.name], author)
    send(ir2)
    return out

def reducemax(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    ir2=DeepxIR("reducemax",[a.node.name,dim,keepdim], [out.node.name], author)
    send(ir2)
    return out

def reducemin(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    ir2=DeepxIR("reducemin",[a.node.name,dim,keepdim], [out.node.name], author)
    send(ir2)
    return out
 