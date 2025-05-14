from typing import Optional, Union
from deepx import Tensor,Shape,Number

from .leaffunc import create_A_B_tf_C,create_A_tf_C,create_A_B_c_tf_D
from .leaffunc_life import newtensor
from .authormap import defaultauthor

#四则运算
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
def rdiv(
        a: Union[float, int],
        b: Tensor, 
        out:Union[Tensor,str]=None)->Tensor:
    outtensor=out
    if isinstance(out,str) or out is None:
        outtensor=newtensor(b.shape,dtype=b.dtype,name=out)
    from .rtf_elementwise import rtf_rdivscalar
    rtf_rdivscalar(a,b,outtensor,defaultauthor['rdivscalar'])
    return outtensor

## 幂、指数 运算 

pow=create_A_B_tf_C('pow')
def rpow(a:Number,b:Tensor,out:Union[Tensor,str]=None)->Tensor:
    outtensor=out
    if isinstance(out,str) or out is None:
        outtensor=newtensor(b.shape,dtype=b.dtype,name=out)
    from .rtf_elementwise import rtf_rpowscalar
    rtf_rpowscalar(a,b,outtensor,defaultauthor['rpowscalar'])
    return outtensor
sqrt=create_A_tf_C('sqrt')
exp=create_A_tf_C('exp')
log=create_A_tf_C('log')

# 三角函数
sin=create_A_tf_C('sin')
cos=create_A_tf_C('cos')
tan=create_A_tf_C('tan')

#取大小值
max=create_A_B_tf_C('max')
min=create_A_B_tf_C('min')

#位运算
invert=create_A_tf_C('invert')

#比较
less=create_A_B_tf_C('less',outtype='bool')
greater=create_A_B_tf_C('greater',outtype='bool')
equal=create_A_B_c_tf_D('equal',outtype='bool')
notequal=create_A_B_c_tf_D('notequal',outtype='bool')

#分支
def switch(X:tuple[Tensor,...], cases:Tensor, out:Union[Tensor,str]=None)->Tensor:
    assert isinstance(X,tuple)
    for x in X:
        assert isinstance(x,Tensor) and x.shape==cases.shape
    outtensor=out
    if isinstance(out,str) or out is None:
        outtensor=newtensor(cases.shape,dtype=X[0].dtype,name=out)
    assert isinstance(outtensor,Tensor) and outtensor.shape==cases.shape

    from .rtf_elementwise import rtf_switch
    rtf_switch(X,cases,outtensor,defaultauthor['switch'])
    return outtensor

#todtype
def todtype(t:Tensor,dest:Tensor):
    assert isinstance(t,Tensor)
    assert isinstance(dest,Tensor)
    assert t.shape==dest.shape

    from .rtf_elementwise import rtf_todtype
    rtf_todtype(t,dest)

