from typing import Union
from deepx.tensor import Tensor,Shape

from .leaffunc_life import newtensor
from .authormap import defaultauthor

def reshape(t:Tensor,shape:list[int],out:Union[Tensor,str]='')->Tensor:
    outtensor=out
    if isinstance(out,str):
        outshape=shape
        outtensor=newtensor(outshape,dtype=t.dtype,name=out)
    else:
        outtensor=out
        outtensor._shape=Shape(shape)
    from .rtf_changeshape import rtf_reshape
    rtf_reshape(t,shape,outtensor,defaultauthor['reshape'])
    return outtensor
 

def permute(t:Tensor,
            dimorder:list[int],
            out:Union[Tensor,str]='')->Tensor:
    if t.ndim!=len(dimorder):
        raise ValueError(f"shape参数不合法,当前输入维度数：{len(dimorder)}，张量维度数：{t.ndim}")
    dimorder = [d % t.ndim for d in dimorder]
    outtensor=out
    if isinstance(out,str):
        outshape = [t.shape[dim] for dim in dimorder]
        outtensor=newtensor(outshape,dtype=t.dtype,name=out)

    from .rtf_changeshape import rtf_transpose
    rtf_transpose(t,dimorder,outtensor,defaultauthor['transpose'])
    return outtensor

def transpose(t:Tensor,out:Union[Tensor,str]='')->Tensor:
    dimorder = list(range(t.ndim))
    dimorder[-1],dimorder[-2]=dimorder[-2],dimorder[-1]
    return permute(t,dimorder,out)

 

def concat(tensors:Union[list[Tensor],tuple[Tensor]],dim:int,out:Union[Tensor,str]='')->Tensor:
    outtensor=out
    if isinstance(out,str):
        outshape=list(tensors[0].shape)
        outshape[dim]=sum(t.shape[dim] for t in tensors)
        outtensor=newtensor(outshape,dtype=tensors[0].dtype,name=out)
    from .rtf_changeshape import rtf_concat
    rtf_concat(tensors,dim,outtensor,defaultauthor['concat'])
    return outtensor

def broadcastTo(t:Tensor,new_shape:tuple[int],out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    if t.shape==new_shape:
        return t
    bshape=Shape.broadcast_shape(t.shape,new_shape)
    if bshape!=new_shape:
        raise ValueError(f"广播失败：{t.shape} 无法广播为 {new_shape} ")
    outtensor=out
    if isinstance(out,str):
        outshape=new_shape
        outtensor=newtensor(outshape,dtype=t.dtype,name=out)
    from .rtf_changeshape import rtf_broadcastTo
    rtf_broadcastTo(t,new_shape,outtensor,defaultauthor['broadcastTo'])
    return outtensor
broadcast_to = broadcastTo

# def unsqueeze(t:Tensor,dim:int)->Tensor:
#     # 确保dim是有效的
#     if dim < -t.ndim-1 or dim > t.ndim:
#         raise ValueError(f"维度超出范围，当前张量维度为{t.ndim},dim={dim}")
    
#     # 处理负数索引
#     if dim < 0:
#         dim = t.ndim + dim + 1

#     new_shape = list(t.shape)
#     new_shape.insert(dim, 1)

#     return reshape(t, new_shape)

# OpNode.register("expand")
# def expand(t:Tensor,shape:list[int],out:Union[Tensor,str]='')->Tensor:
#     outtensor=None
#     if isinstance(out,str):
#         outtensor=Tensor(shape=shape, dtype=t.dtype, device=t.device)
#         outtensor.addtograph(out)
#     else:
#         outtensor=out

#     opnode=t.graph.add_op("expand")
#     opnode.add_input(t.node)
#     opnode.add_input(t.graph.add_vector("",shape))
#     outtensor.node.add_input(opnode)
#     if t.graph.eager:
#         ir=DeepxIR("expand",'',[t.node.name,*map(str, shape)], [outtensor.node.name])
#         send(ir)
#     return outtensor

# def broadcast_to(a: Tensor, shape: tuple,out:Union[Tensor,str]='') -> Tensor:
#     # 计算广播后的形状
#     try:
#         target_shape = broadcast_shape(a.shape, shape)
#         if target_shape!=shape:
#             raise ValueError(f"广播失败：{a.shape} 无法广播为 {shape} ")
#     except ValueError as e:
#         raise ValueError(f"广播失败：{e}") from e
    
#     # 为每个张量添加前导维度
#     if a.shape != target_shape:
#         a_reshape = [1] * (len(target_shape) - a.ndimension) + list(a.shape)
#         a_reshaped =  reshape(a,a_reshape)
#     else:
#         a_reshaped=a
   
#     # 执行实际广播
#     if a_reshaped.shape != target_shape:
#         a_broadcasted =  expand(a_reshaped,target_shape,out)
#     else:
#         a_broadcasted=a_reshaped
    
#     return a_broadcasted 