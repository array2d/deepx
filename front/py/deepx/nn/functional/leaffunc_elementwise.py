from typing import Optional, Union
from deepx import Tensor,Shape

from .leaffunc import create_A_B_tf_C,create_A_tf_C
from .leaffunc_life import newtensor
from .authormap import defaultauthor

# 创建具体操作函数
add = create_A_B_tf_C('add')
sub = create_A_B_tf_C('sub')
mul = create_A_B_tf_C('mul')

#div
def div(
        a: Optional[Union[Tensor, float, int]] = None,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor) and isinstance(a,Tensor):
        #C=A/B
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        an=a
        bn=b
        if a.shape!=b.shape:
            newshape=Shape.broadcast_shape(a.shape,b.shape)
            an=a.broadcastTo(newshape)
            bn=b.broadcastTo(newshape)
        from .rtf_elementwise import rtf_div
        rtf_div(an,bn,outtensor,defaultauthor['div'])
        return outtensor
    else:
        if isinstance(a,Tensor):
            #C=A/b
            outtensor=out
            if isinstance(out,str):
                outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
            from .rtf_elementwise import rtf_divscalar
            rtf_divscalar(a,b,outtensor,defaultauthor['divscalar'])
            return outtensor
        elif isinstance(a,float) or isinstance(a,int):
            #C=a/B
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
#sqrt

sqrt=create_A_tf_C('sqrt')
exp=create_A_tf_C('exp')
log=create_A_tf_C('log')