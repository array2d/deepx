from typing import Union

from deepx import Tensor
from .leaffunc_life import newtensor
from .authormap import defaultauthor

def matmul(a:Tensor,b:Tensor,out:Union[Tensor,str]='')->Tensor:
    outtensor=out
    if isinstance(out,str):
        outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
    from .rtf_matmul import rtf_matmul
    rtf_matmul(a,b,outtensor,defaultauthor['matmul'])
    return outtensor
