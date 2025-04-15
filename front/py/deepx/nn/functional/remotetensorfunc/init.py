from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from typing import Union,Optional

def constant(t:Tensor,value:Union[float,int]=0,author='miaobyte')->Tensor:
    ir=DeepxIR("constant",  [t.node.name,value], [],author)
    send(ir)
    return t
  
def arange(t:Tensor,start:Optional[Union[float,int]]=0,step:Optional[Union[float,int]]=1,author='miaobyte')->Tensor:
    ir=DeepxIR("arange",[t.node.name,start,step], [],author)
    send(ir)
    return t
 
def uniform(t:Tensor,low=0, high=1,seed:int=0,author='miaobyte')->Tensor:
    ir=DeepxIR("uniform",[t.node.name,low, high,seed], [],author)
    send(ir)
    return t