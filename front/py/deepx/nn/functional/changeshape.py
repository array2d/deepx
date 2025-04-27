from deepx import Tensor
from .leaffunc_changeshape import reshape

def squeeze(t:Tensor,dim:int)->Tensor:
    assert isinstance(dim,int)
    assert isinstance(t,Tensor)
    dim=dim%t.ndim
    newshape=list(t.shape)
    newshape.pop(dim)
    return reshape(t,tuple(newshape))

def unsqueeze(t:Tensor,dim:int)->Tensor:
    assert isinstance(dim,int)
    assert isinstance(t,Tensor)
    dim=dim%t.ndim
    newshape=list(t.shape)
    newshape.insert(dim,1)
    return reshape(t,tuple(newshape))