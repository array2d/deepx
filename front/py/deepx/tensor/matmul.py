from typing import Union

from  .tensor import Tensor,tensor_method

@tensor_method
def matmul(self:Tensor,other:Tensor,out:Union[Tensor,str]=''):
    from deepx.nn.functional import matmul as matmul_func
    return matmul_func(self,other,out)