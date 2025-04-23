from deepx.tensor import Tensor,Shape
from .leaffunc_reduce import sum
from .leaffunc_life import newtensor
#mean
 
def mean(a:Tensor,dim:tuple[int,...]=None,keepdim:bool=False)->Tensor:
    assert isinstance(a,Tensor)
    if dim is None:
       dim = list(range(a.ndim))
    else:
        dim=list(dim)
        for i in dim:
            if i < 0:
                dim[i] = i + a.dim()
    total = 1
    for i in dim:
        total *= a.shape[i]
    reduceshape=Shape.reduceshape(a.shape,dim,keepdim)
    out=newtensor(reduceshape,dtype=a.dtype)
    sum(a, tuple(dim), keepdim, out)
    out.div_(total)
    return out
