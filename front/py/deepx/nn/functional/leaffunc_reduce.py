from typing import Optional, Union

from deepx.tensor import Tensor
from deepx.autograd import Function,Context
from .leaffunc_new import newtensor
def reduceshape(inshape: Union[list[int], tuple[int]], 
               dim: Union[list[int], tuple[int]], 
               keepdim: bool) -> tuple[int]:
    """计算维度缩减后的形状
    
    Args:
        inshape: 输入形状，如(2,3,4)
        dim: 要缩减的维度列表，支持负数索引，如[-1]
        keepdim: 是否保留缩减后的维度为1
        
    Returns:
        缩减后的形状元组
        
    Example:
        >>> reduceshape((2,3,4), [1], True)
        (2, 1, 4)
        >>> reduceshape((2,3,4), [1], False)
        (2, 4)
    """
    ndim = len(inshape)
    # 处理负数维度
    normalized_dim = [d % ndim for d in dim]
    # 去重并排序
    unique_dim = sorted(set(normalized_dim))
    
    if keepdim:
        return tuple(1 if i in unique_dim else s 
                   for i, s in enumerate(inshape))
    else:
        return tuple(s for i, s in enumerate(inshape)
                   if i not in unique_dim)
 
#sum    
 
class Sum(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor,dims:tuple[int]=None,keepdim:bool=False,out:Union[Tensor,str]='',authormap:dict={'sum':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_tensors('a',a)
            ctx.save_data('dims',dims)
            ctx.save_data('keepdim',keepdim)
        ctx.set_authormap(authormap)
        if dims is None:
            dims=tuple(range(a.ndim))

        result=out
        if isinstance(out,str):
            resultshape=reduceshape(a.shape,dims,keepdim)
            result=newtensor(resultshape, dtype=a.dtype,name=out)
        from .rtf_reduce import rtf_sum
        rtf_sum(a,dims,keepdim,result,ctx.authormap['sum'])
        return result
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        pass
    
    

def sum(
        a:Tensor,
        dims:list[int],
        keepdim:bool=False,
        out:Union[Tensor,str]='',
        author:str='miaobyte',
        requires_grad:bool=False)->Tensor:
    return Sum.apply(a,dims,keepdim,out,{'sum':author},requires_grad=requires_grad)

#prod
  
class Prod(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor,dims:tuple[int]=None,keepdim:bool=False,out:Union[Tensor,str]='',authormap:dict={'prod':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_tensors('a',a)
            ctx.save_data('dims',dims)
            ctx.save_data('keepdim',keepdim)
        ctx.set_authormap(authormap)
        if dims is None:
            dims=tuple(range(a.ndim))

        result=out
        if isinstance(out,str):
            resultshape=reduceshape(a.shape,dims,keepdim)
            result=newtensor(resultshape, dtype=a.dtype,name=out)
        from .rtf_reduce import rtf_prod
        rtf_prod(a,dims,keepdim,result,ctx.authormap['prod'])
        return result
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        pass

def prod(
        a:Tensor,
        dims:list[int],
        keepdim:bool=False,
        out:Union[Tensor,str]='',
        author:str='miaobyte',
        requires_grad:bool=False)->Tensor:
    return Prod.apply(a,dims,keepdim,out,{'prod':author},requires_grad=requires_grad)

#max
  
class ReduceMax(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor,dims:tuple[int]=None,keepdim:bool=False,out:Union[Tensor,str]='',authormap:dict={'reducemax':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_tensors('a',a)
            ctx.save_data('dims',dims)
            ctx.save_data('keepdim',keepdim)
        ctx.set_authormap(authormap)
        if dims is None:
            dims=tuple(range(a.ndim))

        result=out
        if isinstance(out,str):
            resultshape=reduceshape(a.shape,dims,keepdim)
            result=newtensor(resultshape, dtype=a.dtype,name=out)
        from .rtf_reduce import rtf_reducemax
        rtf_reducemax(a,dims,keepdim,result,ctx.authormap['reducemax'])
        return result
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        pass
    
    

def reducemax(
        a:Tensor,
        dims:list[int],
        keepdim:bool=False,
        out:Union[Tensor,str]='',
        author:str='miaobyte',
        requires_grad:bool=False)->Tensor:
    return ReduceMax.apply(a,dims,keepdim,out,{'reducemax':author},requires_grad=requires_grad)
 
#min    
class ReduceMin(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor,dims:tuple[int]=None,keepdim:bool=False,out:Union[Tensor,str]='',authormap:dict={'reducemin':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_tensors('a',a)
            ctx.save_data('dims',dims)
            ctx.save_data('keepdim',keepdim)
        ctx.set_authormap(authormap)
        if dims is None:
            dims=tuple(range(a.ndim))

        result=out
        if isinstance(out,str):
            resultshape=reduceshape(a.shape,dims,keepdim)
            result=newtensor(resultshape, dtype=a.dtype,name=out)
        from .rtf_reduce import rtf_reducemin
        rtf_reducemin(a,dims,keepdim,result,ctx.authormap['reducemin'])
        return result
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        pass
    
    

def reducemin(
        a:Tensor,
        dims:list[int],
        keepdim:bool=False,
        out:Union[Tensor,str]='',
        author:str='miaobyte',
        requires_grad:bool=False)->Tensor:
    return ReduceMin.apply(a,dims,keepdim,out,{'reducemin':author},requires_grad=requires_grad)
 
 