from deepx.tensor import Tensor
from typing import Union

def rsqrt(input:Tensor,out:Union[Tensor,str]=None,requires_grad:bool=False)->Tensor:
    from .leaffunc_elementwise import sqrt,div
    out=sqrt(input, out, requires_grad)
    return div(1,out,out,requires_grad)


