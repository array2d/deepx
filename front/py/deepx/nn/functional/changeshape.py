from typing import Union
from deepx import Tensor
from .leaffunc_changeshape import reshape,indexselect, concat,broadcastTo
from .leaffunc_init import newtensor,arange
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
    dim = dim % (t.ndim + 1)
    newshape=list(t.shape)
    newshape.insert(dim,1)
    return reshape(t,tuple(newshape))

def sliceselect(t:Tensor,sliceobj:slice,dim:int=-1,out:Union[Tensor,str]='')->Tensor:
    assert isinstance(dim,int)
    assert isinstance(sliceobj,slice)
    assert isinstance(t,Tensor)
    dim=dim%t.ndim
    start=start = 0 if sliceobj.start is None else sliceobj.start % t.shape[dim]
    stop= t.shape[dim] if sliceobj.stop is None else sliceobj.stop % t.shape[dim]
    
    index=arange(start,stop,dtype='int32')
    return  indexselect(t,index,dim=dim,out=out)

cat= concat
# 参考 PyTorch 文档，broadcastTo和expand是作用一样
#  https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html
expand = broadcastTo