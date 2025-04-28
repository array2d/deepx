from deepx.tensor import Tensor
from deepx.nn.functional import newtensor

def rsqrt(input:Tensor)->Tensor:
    from .leaffunc_elementwise import sqrt
    return 1/sqrt(input)
 