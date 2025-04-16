from typing import Optional,Union

from  .tensor import Tensor,tensor_method

@tensor_method
def matmul(self,other,out:Union[Tensor,str]='',author='miaobyte'):
    from deepx.nn.functional import matmul as matmul_func
    return matmul_func(self,other,out,author)