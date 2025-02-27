from typing import Optional

from .tensor import Tensor, tensor_method
import numpy as np
from .deepxir import DeepxIR
from deepx.scheduler import send
def full(*shape, fill_value=0, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    t=Tensor(data=None, shape=shape, dtype=dtype, device=device)
    if t.graph.eager:
        ir=DeepxIR("constant", t.dtype, [fill_value], [t.node.name])
        send(str(ir))
    return t

def zeros(*shape, dtype=None, device=None):
    return full(*shape, fill_value=0, dtype=dtype, device=device)

def ones(*size, dtype=None, device=None):
    return full(*size, fill_value=1, dtype=dtype, device=device)

def rand(*size, dtype=None, device=None):
   #TODO
   pass

def randn(*size, dtype=None, device=None):
    #TODO
    pass

def arange(start, end=None, step=1, dtype=None, device=None):
    #TODO
    pass

def eye(
        n:int,
        m:Optional[int]=None,
        dtype:Optional[str]=None, 
        device:Optional[str]=None):
    #TODO
    pass
 
