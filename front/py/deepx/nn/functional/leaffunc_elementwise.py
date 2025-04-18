from typing import Optional, Union
from deepx import Tensor,Shape,Number

from .leaffunc import create_A_B_tf_C,create_A_tf_C
from .leaffunc_life import newtensor
from .authormap import defaultauthor

# 创建具体操作函数
add = create_A_B_tf_C('add')
sub = create_A_B_tf_C('sub')
mul = create_A_B_tf_C('mul')
_div=create_A_B_tf_C('div')

def div(
        a: Union[Tensor, float, int],
        b: Union[Tensor, float, int], 
        out:Union[Tensor,str]=None)->Tensor:
    if isinstance(a,Tensor):
        return _div(a,b,out)
    elif isinstance(a,float) or isinstance(a,int):
        return rdiv(a,b,out)
    else:
        raise ValueError(f"Invalid type for a: {type(a)}")

#div
def rdiv(
        a: Union[float, int],
        b: Tensor, 
        out:Union[Tensor,str]=None)->Tensor:
    outtensor=out
    if isinstance(out,str):
        outtensor=newtensor(b.shape,dtype=b.dtype,name=out)
    from .rtf_elementwise import rtf_rdivscalar
    rtf_rdivscalar(a,b,outtensor,defaultauthor['rdivscalar'])
    return outtensor
 
max=create_A_B_tf_C('max')
min=create_A_B_tf_C('min')

#pow
pow=create_A_B_tf_C('pow')
def rpow(a:Number,b:Tensor,out:Union[Tensor,str]=None)->Tensor:
    outtensor=out
    if isinstance(out,str):
        outtensor=newtensor(b.shape,dtype=b.dtype,name=out)
    from .rtf_elementwise import rtf_rpowscalar
    rtf_rpowscalar(a,b,outtensor,defaultauthor['rpowscalar'])
    return outtensor
#sqrt

sqrt=create_A_tf_C('sqrt')
exp=create_A_tf_C('exp')
log=create_A_tf_C('log')

#invert
invert=create_A_tf_C('invert')