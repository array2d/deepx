from deepx.tensor import Tensor,Shape
from typing import Optional,Union
from .leaffunc_reduce import sum
from .leaffunc_life import newtensor
#mean
 
def mean(a:Tensor,dim:tuple[int,...]=None,keepdim:bool=False)->Tensor:
    # 如果dim为None,则对所有维度求平均
    if dim is None:
        dim = list(range(a.ndim))
    dim=list(dim)
    total = 1
    for i in dim:
        if i < 0:
            dim[i] = i + a.dim()
        total *= a.shape[i]
    reduceshape=Shape.reduceshape(a.shape,dim,keepdim)
    out=newtensor(reduceshape,dtype=a.dtype)
    sum(a, tuple(dim), keepdim, out)
    out.div_(total)
    return out
