from typing import Union,Tuple
from deepx.tensor import Tensor,Shape
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import  Function,Context

from .leaffunc_new import newtensor

class Reshape(Function):
    @staticmethod
    def forward(ctx:Context,t:Tensor,shape:list[int],out:Union[Tensor,str],authormap:dict):
        if ctx.requires_grad:
            ctx.save_data('oldshape',t.shape)
            ctx.save_tensors('t',t)
        ctx.set_authormap(authormap)

        total_size=1
        for i in shape:
            total_size*=i
        if total_size!=t.numel():
            raise ValueError(f"reshape失败：{t.shape} 无法reshape为 {shape} ")
        outtensor=out
        if isinstance(out,str):
            outshape=shape
            outtensor=newtensor(outshape,dtype=t.dtype,name=out)
        else:
            outtensor=out
            outtensor._shape=Shape(shape)
        from .rtf_changeshape import rtf_reshape
        rtf_reshape(t,shape,outtensor,ctx.authormap['reshape'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,t_grad:Tensor,out_grad:Tensor):
        oldshape=ctx.get_data('oldshape')
        t=ctx.get_tensor('t')
        from .rtf_changeshape import rtf_reshape
        rtf_reshape(out_grad,oldshape,t_grad,ctx.authormap['reshape'])
        return t_grad

def reshape(t:Tensor,shape:list[int],out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    return Reshape.apply(t,shape,out,{'reshape':author},requires_grad=requires_grad)

 
class Permute(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                dimorder:list[int],
                out:Union[Tensor,str]='',
                authormap:dict={'transpose':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_data('dimorder',dimorder)
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outshape = [t.shape[dim] for dim in dimorder]
            outtensor=newtensor(outshape,dtype=t.dtype,name=out)

        from .rtf_changeshape import rtf_transpose
        rtf_transpose(t,dimorder,outtensor,ctx.authormap['transpose'])
        return outtensor
        
    
    @staticmethod
    def backward(ctx:Context,in_grad,out_grad):
        dimorder=ctx.get_data('dimorder')
        inverse_dimorder = [0] * len(dimorder)
        for i, j in enumerate(dimorder):
            inverse_dimorder[j] = i
        from .rtf_changeshape import rtf_transpose
        rtf_transpose(out_grad,inverse_dimorder,in_grad,ctx.authormap['transpose'])
        return in_grad

def permute(t:Tensor,
            dimorder:list[int],
            out:Union[Tensor,str]='',
            requires_grad:bool=False,
            author='miaobyte')->Tensor:
    if t.dim!=len(dimorder):
        raise ValueError(f"shape参数不合法,当前输入维度数：{len(dimorder)}，张量维度数：{t.dim}")
    dimorder = [d % t.ndim for d in dimorder]
    return Permute.apply(t,dimorder,out,{'transpose':author},requires_grad=requires_grad)
 
def transpose(t:Tensor,out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    dimorder = list(range(t.ndim))
    dimorder[-1],dimorder[-2]=dimorder[-2],dimorder[-1]
    return Permute.apply(t,dimorder,out,{'transpose':author},requires_grad=requires_grad)


 
class Concat(Function):
    @staticmethod
    def forward(ctx:Context,
                tensors:list[Tensor],
                dim:int,
                out:Union[Tensor,str]='',
                authormap:dict={'concat':'miaobyte'})->Tensor:
        if ctx.requires_grad:
            ctx.save_data('dim',dim)
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outshape=list(tensors[0].shape)
            outshape[dim]=sum(t.shape[dim] for t in tensors)
            outtensor=newtensor(outshape,dtype=tensors[0].dtype,name=out)
        from .rtf_changeshape import rtf_concat
        rtf_concat(tensors,dim,outtensor,ctx.authormap['concat'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        dim=ctx.get_data('dim')
        #todo: 反向传播

def concat(t:Tensor,dim:int,out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    return Concat.apply(t,dim,out,{"concat":author},requires_grad=requires_grad)

def broadcast_shape(shape_a: tuple[int], shape_b: tuple[int]) -> tuple[int]:
    """计算两个形状的广播后形状"""
    # 获取形状的长度
    len_a, len_b = len(shape_a), len(shape_b)
    
    # 创建结果形状
    result_shape = []
    
    # 从右往左对齐并计算每个维度
    for i in range(1, min(len_a, len_b) + 1):
        dim_a = shape_a[-i]
        dim_b = shape_b[-i]
        
        if dim_a == 1 or dim_b == 1:
            # 广播规则：如果一个维度为1，取另一个维度的值
            result_shape.insert(0, max(dim_a, dim_b))
        elif dim_a == dim_b:
            # 维度相同，保持不变
            result_shape.insert(0, dim_a)
        else:
            # 维度不同且都不为1，无法广播
            raise ValueError(f"无法广播的形状：{shape_a} 和 {shape_b}")
    
    # 添加较长形状中多出的前导维度
    if len_a > len_b:
        result_shape = list(shape_a[:len_a - len_b]) + result_shape
    elif len_b > len_a:
        result_shape = list(shape_b[:len_b - len_a]) + result_shape
    
    return tuple(result_shape)
 
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                new_shape:tuple[int],
                out:Union[Tensor,str]='',
                authormap:dict={'broadcastTo':'miaobyte'})->Tensor:
        bshape=broadcast_shape(t.shape,new_shape)
        if bshape!=new_shape:
            raise ValueError(f"广播失败：{t.shape} 无法广播为 {new_shape} ")
        
        if ctx.requires_grad:
            ctx.save_data('new_shape',new_shape)
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outshape=new_shape
            outtensor=newtensor(outshape,dtype=t.dtype,name=out)
        from .rtf_changeshape import rtf_broadcastTo
        rtf_broadcastTo(t,new_shape,outtensor,ctx.authormap['broadcastTo'])
        return outtensor
    
    #todo: 反向传播
    @staticmethod
    def backward(ctx:Context,out_grad):
        new_shape=ctx.get_data('new_shape')
        #todo: 反向传播

def broadcastTo(t:Tensor,new_shape:tuple[int],out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    return BroadcastTo.apply(t,new_shape,out,{'broadcastTo':author},requires_grad=requires_grad)

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