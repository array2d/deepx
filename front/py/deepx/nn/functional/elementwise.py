from deepx.tensor import Tensor
from deepx.nn.functional import newtensor

def rsqrt(input:Tensor)->Tensor:
    from .leaffunc_elementwise import sqrt,div
    outtensor=input
    if input.name is not None:
        outtensor=newtensor(input.shape, dtype=input.dtype)
    sqrt(input,out= outtensor)
    return div(1,outtensor,outtensor)
 

