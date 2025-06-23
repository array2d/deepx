from typing import Union
from  .tensor import Tensor,tensor_method

@tensor_method
def reshape(self,shape:tuple[int,...],out:Union[Tensor,str]='')->Tensor:
    assert isinstance(shape,tuple)
    from deepx.nn.functional import reshape as reshape_func
    result=reshape_func(self,shape,out)
    return result

@tensor_method
def reshape_(self,shape:tuple[int,...])->Tensor:
    assert isinstance(shape,tuple)
    from deepx.nn.functional import reshape as reshape_func
    result=reshape_func(self,shape,self)
    return result

@tensor_method
def permute(self,dimorder:tuple[int,...],out:Union[Tensor,str]=''):
    assert isinstance(dimorder,tuple)
    from deepx.nn.functional import permute as permute_func
    result=permute_func(self,dimorder,out)
    return result

@tensor_method
def permute_(self,dimorder:tuple[int,...])->Tensor:
    assert isinstance(dimorder,tuple)
    from deepx.nn.functional import permute as permute_func
    permute_func(self,dimorder,self)
    return self

@tensor_method
def transpose(self,out:Union[Tensor,str]=''):
    assert isinstance(out,str) or isinstance(out,Tensor)
    from deepx.nn.functional import transpose as transpose_func
    result=transpose_func(self,out)
    return result

@tensor_method
def transpose_(self):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,self)
    return self

# broadcast_to==broadcastTo==expand
# https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html
@tensor_method
def broadcastTo(self,shape:tuple[int,...],out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import broadcastTo as broadcastTo_func
    result=broadcastTo_func(self,shape,out)
    return result

@tensor_method
def broadcast_to(self,shape:tuple[int,...],out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import broadcastTo as broadcast_to_func
    result=broadcast_to_func(self,shape,out)
    return result

@tensor_method
def expand(self,shape:tuple[int,...],out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import broadcastTo as expand_func
    result=expand_func(self,shape,out)
    return result

@tensor_method
def indexselect(self,index:Tensor,gatheraxis:int=0,out:Union[Tensor,str]='')->Tensor:
    assert isinstance(index,Tensor)
    gatheraxis=gatheraxis%self.ndim
    from deepx.nn.functional import indexselect as indexselect_func
    result=indexselect_func(self,index,gatheraxis,out)
    return result

@tensor_method
def sliceselect(self,index:slice,dim:int=0,out:Union[Tensor,str]='')->Tensor:
    assert isinstance(index,slice)
    gatheraxis=dim%self.ndim
    from deepx.nn.functional import sliceselect as sliceselect_func
    result=sliceselect_func(self,index,gatheraxis,out)
    return result

@tensor_method
def squeeze(self,dim:int)->Tensor:
    from deepx.nn.functional import squeeze as squeeze_func
    result=squeeze_func(self,dim)
    return result

@tensor_method
def unsqueeze(self,dim:int)->Tensor:
    from deepx.nn.functional import unsqueeze as unsqueeze_func
    result=unsqueeze_func(self,dim)
    return result

@tensor_method
def repeat(self,repeats:tuple[int,...])->Tensor:
    from deepx.nn.functional import repeat as repeat_func
    result=repeat_func(self,repeats)
    return result

