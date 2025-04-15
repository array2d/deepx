from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def rtf_reshape(t:Tensor,shape:tuple[int],out:Tensor,author='miaobyte'):
    args=[Param.tensor(t),Param.vector(shape,'int32')]
    returns=[Param.tensor(out)]
    ir=DeepxIR("reshape", args, returns,author)
    send(ir)


def rtf_transpose(t:Tensor,dimorder:tuple[int],out:Tensor,author='miaobyte'):
    args=[Param.tensor(t),Param.vector(dimorder,'int32')]
    returns=[Param.tensor(out)]
    ir=DeepxIR("transpose", args, returns,author)
    send(ir)
 
def rtf_concat(tensors:tuple[Tensor],dim:int,out:Tensor,author='miaobyte'):
    args=[Param.tensorlist([t for t in tensors],tensors[0].dtype),Param.varnum(dim,'int32')]
    returns=[Param.tensor(out)]
    ir=DeepxIR("concat", args, returns,author)
    send(ir)
 

def rtf_broadcastTo(t:Tensor,new_shape:tuple[int],out:Tensor,author='miaobyte'):
    args=[Param.tensor(t),Param.vector(new_shape,'int32')]
    returns=[Param.tensor(out)]
    ir=DeepxIR("broadcastTo", args, returns,author)
    send(ir)
 
