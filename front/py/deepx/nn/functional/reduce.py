from typing import Optional, Union

from deepx.tensor import Tensor
from deepx.autograd.graph import OpNode
from deepx.nn.deepxir import DeepxIR    
from deepx.scheduler import send
from .elementwise import _A_b_elementwiseop_C

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

def _A_v_reduceop_C(
        a:Tensor,
        dim: Union[list[int],tuple[int]] = [],
        keepdim:bool=False,
        op:str=None,
        out:Union[Tensor,str]='')->Tensor:
    
    if dim is None:
        dim=list(range(a.ndim))

    result=None
    if isinstance(out,str):
        resultshape=reduceshape(a.shape,dim,keepdim)
        result=Tensor(shape=resultshape, dtype=a.dtype, device=a.device)
        result.addtograph(out)
    else:
        result=out

    vector_node=a.graph.add_vector("",dim)
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(vector_node)
    result.node.add_input(opnode)

    if a.graph.eager:
        args = [*dim, "keepdim"] if keepdim else [*dim]
        varir=DeepxIR("argset",'int32', args, [vector_node.name])

        send(varir)
        ir=DeepxIR(op, a.dtype, [a.node.name,vector_node.name], [result.node.name])
        send(ir)
    return result

#max

OpNode.register("reduce_max")
def reduce_max(
        a:Tensor,
        dim:list[int] = None,
        keepdim=False,
        out:Union[Tensor,str]='')->Tensor:
    return _A_v_reduceop_C(a,dim,keepdim,"max",out)
 
#min    
OpNode.register("reduce_min")
def reduce_min(
        a:Tensor,
        dim:list[int] = None,
        keepdim=False,
        out:Union[Tensor,str]='')->Tensor:
    return _A_v_reduceop_C(a,dim,keepdim,"min",out)
    
 
#sum    
OpNode.register("sum")
def sum(
        a:Tensor,
        dim:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        keepdim:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    return _A_v_reduceop_C(a,dim,keepdim,"sum",out)

#prod
OpNode.register("prod")
def prod(
        a:Tensor,
        dim:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        keepdim:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    return _A_v_reduceop_C(a,dim,keepdim,"prod",out)

#mean
OpNode.register("mean")
def mean(
        a:Tensor,
        dim:Optional[Union[list[int],tuple[int]]]=None,
        keepdim:bool=False,
        out:Union[str]='')->Tensor:
    # 如果dim为None,则对所有维度求平均
    if dim is None:
        dim = list(range(a.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    total = 1
    for i in dim:
        if i < 0:
            dim[i] = i + a.dim()
        total *= a.shape[i]
    result = sum(a, dim, keepdim, out)/total
    return result
# #var
# OpNode.register("var")