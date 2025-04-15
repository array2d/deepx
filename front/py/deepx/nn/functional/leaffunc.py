from typing import Union,Tuple
from deepx.tensor import Tensor,Shape
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode,Function,Context

def buildgraph(a:Tensor,dim:tuple[int],keepdim:bool=False,out:Tensor, author:str='miaobyte')->Tensor: