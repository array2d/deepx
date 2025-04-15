from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send
from typing import Union,Optional
from .rtf import A_B_op_C,A_scalar_op_C,A_op_C


def rtf_constant(t:Tensor,value:Union[float,int]=0,author='miaobyte')->Tensor:
    A_scalar_op_C("constant",t,value,t,author)
    return t
  
def rtf_arange(t:Tensor,start:Optional[Union[float,int]]=0,step:Optional[Union[float,int]]=1,author='miaobyte')->Tensor:
    args=[Param.tensor(t),Param.varnum(start),Param.varnum(step)]
    returns=[Param.tensor(t)]
    ir=DeepxIR("arange", args, returns,author)
    send(ir)
    return t
 
def rtf_uniform(t:Tensor,low=0, high=1,seed:int=0,author='miaobyte')->Tensor:
    args=[Param.tensor(t),Param.varnum(low),Param.varnum(high),Param.varnum(seed)]
    returns=[Param.tensor(t)]
    ir=DeepxIR("uniform", args, returns,author)
    send(ir)
    return t