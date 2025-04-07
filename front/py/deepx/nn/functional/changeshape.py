from typing import Union,Tuple
from deepx.tensor import Tensor,Shape
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode,Function,Context

def _A_v_elementwiseop_C(
        a:Tensor,
        b: list[int] ,
        op:str=None,
        out:Tensor=None,
        author='miaobyte')->Tensor:
    g=a.graph
    opnode = g.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(g.add_vector("",b))

    outtensor=out
    outtensor.node.add_input(opnode)
    if g.eager:
        ir=DeepxIR(op, [a.node.name,b], [outtensor.node.name],author)
        send(ir)
    return outtensor

OpNode.register("reshape")
class Reshape(Function):
    @staticmethod
    def forward(ctx:Context,t:Tensor,shape:list[int],out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('oldshape',t.shape)
            ctx.save_tensors('t',t)
        outtensor=out
        if isinstance(out,str):
            outshape=shape
            outtensor=Tensor(shape=outshape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        outtensor._shape=Shape(shape)
        return _A_v_elementwiseop_C(t,shape,"reshape",outtensor,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        oldshape=ctx.get_data('oldshape')
        t=ctx.get_tensor('t')
        return _A_v_elementwiseop_C(out_grad,oldshape,"reshape",t.node.name,author)

def reshape(t:Tensor,shape:list[int],out:Union[Tensor,str]='',author='miaobyte',requires_grad:bool=False)->Tensor:
    if t.shape==shape:
        return t
    return Reshape.apply(t,shape,out,author,requires_grad=requires_grad)


OpNode.register("transpose")
class Permute(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                dimorder:list[int],
                out:Union[Tensor,str]='',
                author='miaobyte')->Tensor:
        if ctx.requires_grad:
            ctx.save_data('dimorder',dimorder)
        outtensor=out
        if isinstance(out,str):
            outshape = [t.shape[dim] for dim in dimorder]
            outtensor=Tensor(shape=outshape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        return _A_v_elementwiseop_C(t,dimorder,"transpose",outtensor,author)
    
    @staticmethod
    def backward(ctx:Context,in_grad,out_grad,author='miaobyte'):
        dimorder=ctx.get_data('dimorder')
        inverse_dimorder = [0] * len(dimorder)
        for i, j in enumerate(dimorder):
            inverse_dimorder[j] = i
        return _A_v_elementwiseop_C(out_grad,inverse_dimorder,"transpose",in_grad,author)

def permute(t:Tensor,
            dimorder:list[int],
            out:Union[Tensor,str]='',
            requires_grad:bool=False,
            author='miaobyte')->Tensor:
    if t.dim!=len(dimorder):
        raise ValueError(f"shape参数不合法,当前输入维度数：{len(dimorder)}，张量维度数：{t.dim}")
    dimorder = [d % t.ndim for d in dimorder]
    return Permute.apply(t,dimorder,out,requires_grad=requires_grad)
 
def transpose(t:Tensor,out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    dimorder = list(range(t.ndim))
    dimorder[-1],dimorder[-2]=dimorder[-2],dimorder[-1]
    return Permute.apply(t,dimorder,out,author,requires_grad=requires_grad)




OpNode.register("concat")
class Concat(Function):
    @staticmethod
    def forward(ctx:Context,
                tensors:list[Tensor],
                dim:int,
                out:Union[Tensor,str]='',
                author='miaobyte')->Tensor:
        if ctx.requires_grad:
            ctx.save_data('dim',dim)
        outtensor=out
        if isinstance(out,str):
            outshape=list(tensors[0].shape)
            outshape[dim]=sum(t.shape[dim] for t in tensors)
            outtensor=Tensor(shape=outshape, dtype=tensors[0].dtype, device=tensors[0].device)
            outtensor.addtograph(out)

        g=tensors[0].graph
        opnode = g.add_op("concat")
        for t in tensors:
            opnode.add_input(t.node)
        opnode.add_input(g.add_var("",dim))

        outtensor.node.add_input(opnode)
        if g.eager:
            ir=DeepxIR("concat", [[t.node.name for t in tensors], dim], [outtensor.node.name],author)
            send(ir)
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad,author='miaobyte'):
        dim=ctx.get_data('dim')
        return _A_v_elementwiseop_C(out_grad,dim,"concat",t.node.name,author)

def concat(t:Tensor,dim:int,out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    return Concat.apply(t,dim,out,author,requires_grad=requires_grad)

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

OpNode.register("broadcastTo")
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                new_shape:tuple[int],
                out:Union[Tensor,str]='',author='miaobyte')->Tensor:
        bshape=broadcast_shape(t.shape,new_shape)
        if bshape!=new_shape:
            raise ValueError(f"广播失败：{t.shape} 无法广播为 {new_shape} ")
        
        if ctx.requires_grad:
            ctx.save_data('new_shape',new_shape)
        outtensor=out
        if isinstance(out,str):
            outshape=new_shape
            outtensor=Tensor(shape=outshape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        return _A_v_elementwiseop_C(t,new_shape,"broadcastTo",outtensor,author)
    
    #todo: 反向传播
    @staticmethod
    def backward(ctx:Context,out_grad,author='miaobyte'):
        new_shape=ctx.get_data('new_shape')
        return _A_v_elementwiseop_C(out_grad,new_shape,"broadcastTo",t.node.name,author)

def broadcast_to(t:Tensor,new_shape:tuple[int],out:Union[Tensor,str]='',requires_grad:bool=False,author='miaobyte')->Tensor:
    return BroadcastTo.apply(t,new_shape,out,author,requires_grad=requires_grad)
    

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