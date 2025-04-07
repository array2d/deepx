from typing import Union
from  .tensor import Tensor,tensor_method

@tensor_method
def reshape(self,*shape,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import reshape as reshape_func
    result=reshape_func(self,shape,out)
    return result

@tensor_method
def reshape_(self,*shape)->Tensor:
    from deepx.nn.functional import reshape as reshape_func
    result=reshape_func(self,shape,self)
    return result

@tensor_method
def permute(self,*axes,out:Union[Tensor,str]=''):
    from deepx.nn.functional import permute as permute_func
    result=permute_func(self,axes,out)
    return result

@tensor_method
def permute_(self,*axes):
    from deepx.nn.functional import permute as permute_func
    permute_func(self,axes,self)
    return self

@tensor_method
def transpose(self,out:Union[Tensor,str]=''):
    from deepx.nn.functional import transpose as transpose_func
    result=transpose_func(self,out)
    return result

@tensor_method
def transpose_(self):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,self)
    return self

@tensor_method
def broadcastshape(self,other:Tensor)->tuple[int]:
    from deepx.nn.functional import broadcastshape as broadcastshape_func
    result=broadcastshape_func(self.shape,other.shape)
    return result

@tensor_method
def broadcast_to(self,shape:tuple[int],out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import broadcast_to as broadcast_to_func
    result=broadcast_to_func(self,shape,out)
    return result



# @tensor_method
# def expand(self,shape:tuple)->Tensor:
#     from deepx.nn.functional import expand as expand_func
#     result=expand_func(self,shape,False)
#     return result
