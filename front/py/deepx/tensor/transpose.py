from typing import Union
from  .tensor import Tensor,tensor_method

@tensor_method
def transpose(self,*axes,out:Union[Tensor,str]=''):
    from deepx.nn.functional import transpose as transpose_func
    result=transpose_func(self,axes,False,out)
    return result

@tensor_method
def transpose_(self,*axes):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,axes,self)
    return self

@tensor_method
def reshape(self,*shape,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import reshape as reshape_func   
    result=reshape_func(self,shape,False,out)
    return result

@tensor_method
def reshape_(self,*shape)->Tensor:
    from deepx.nn.functional import reshape as reshape_func   
    result=reshape_func(self,shape,True)
    return result
