from typing import Union
import importlib


from deepx.tensor import Tensor,Shape
from .leaffunc_life import newtensor
from .authormap import defaultauthor

# inplace操作的函数，其名为_后缀, 返回值为空
# 非inplace操作的函数，其名为_后缀, 返回值为Tensor


def create_A_B_tf_C(op_name):
    """创建元素级操作函数"""
    def op_func(
            a: Tensor,
            b: Union[Tensor, float, int] = None, 
            out: Union[Tensor, str] = None) -> Tensor:
        outtensor = out
        rtf_module = importlib.import_module('deepx.nn.functional.rtf_elementwise')
        if isinstance(b, Tensor):
            an=a
            bn=b
            if a.shape != b.shape:
                newshape = Shape.broadcast_shape(a.shape, b.shape)
                an = a.broadcastTo(newshape)
                bn = b.broadcastTo(newshape)
                if isinstance(out,str):
                    outtensor=newtensor(newshape,dtype=a.dtype,name=out)
            else:
                if isinstance(out,str):
                    outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
            rtf_func = getattr(rtf_module, f'rtf_{op_name}')
            rtf_func(an, bn, outtensor, defaultauthor[op_name])
        else:
            if isinstance(out,str):
                outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
            rtf_func = getattr(rtf_module, f'rtf_{op_name}scalar')
            rtf_func(a, b, outtensor, defaultauthor[f'{op_name}scalar'])
        return outtensor
    op_func.__name__ = op_name
    op_func.__qualname__ = op_name
    return op_func

def create_A_tf_C(op_name):
    def op_func(
            a:Tensor,
            out:Union[Tensor,str]=None)->Tensor:
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        rtf_module = importlib.import_module('deepx.nn.functional.rtf_elementwise')
        rtf_func = getattr(rtf_module, f'rtf_{op_name}')
        rtf_func(a,outtensor,defaultauthor[op_name])
        return outtensor
    op_func.__name__ = op_name
    op_func.__qualname__ = op_name
    return op_func
 

def create_A_dim_keepdim_tf_C(op_name):
    def op_func(
            a:Tensor,
            dim:tuple[int,...],
            keepdim:bool=False,
            out:Union[Tensor,str]='',
            author:str='miaobyte',
            requires_grad:bool=False)->Tensor:
        if dim is None:
            dim=tuple(range(a.ndim))
        result=out
        if isinstance(out,str):
            resultshape=Shape.reduceshape(a.shape,dim,keepdim)
            result=newtensor(resultshape, dtype=a.dtype,name=out)
        rtf_module = importlib.import_module('deepx.nn.functional.rtf_reduce')
        rtf_func = getattr(rtf_module, f'rtf_{op_name}')
        rtf_func(a, dim, keepdim, result, author)
        return result
    return op_func