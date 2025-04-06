from typing import Optional, Union
from deepx import Tensor
from deepx.autograd import Graph,DataNode,OpNode,Function,Context
from deepx.nn import DeepxIR,Param
from deepx.scheduler import send

def _A_elementwiseop_C(
        a:Tensor,
        op:str=None,
        out:Union[Tensor,str]="",author='miaobyte')->Tensor:
    g=a.graph
 
    opnode = g.add_op(op)
    opnode.add_input(a.node)
    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
        outtensor.addtograph(out)
    else:
        outtensor=out   
    outtensor.node.add_input(opnode)
    if g.eager:
        ir=DeepxIR(op, [a.node.name], [outtensor.node.name],author)
        send(ir)
    return outtensor

def _A_B_elementwiseop_C(
        a:Tensor,
        b: Tensor, 
        op:str=None,
        out:Union[Tensor,str]="",author='miaobyte')->Tensor:
    g=a.graph
    if g is None:
       g=b.graph

    A,B=a,b
    if a.shape != b.shape:
        broadcastshape=broadcast_shape(a.shape,b.shape)
        from .changeshape import broadcast_to
        if a.shape != broadcastshape:
            A=broadcast_to(a,broadcastshape)
        if b.shape != broadcastshape:
            B= broadcast_to(b,broadcastshape)

    opnode = g.add_op(op)
    opnode.add_input(A.node)
    opnode.add_input(B.node)
    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=A.shape, dtype=A.dtype, device=A.device)
        outtensor.addtograph(out)
    else:
        outtensor=out   
    outtensor.node.add_input(opnode)
    if g.eager:
        ir=DeepxIR(op, [A.node.name, B.node.name], [outtensor.node.name],author)
        send(ir)
    return outtensor
def _A_b_elementwiseop_C(
        a:Tensor,
        b: Union[ float, int] ,
        op:str=None,
        out:Union[Tensor,str]="",author='miaobyte')->Tensor:
    g=a.graph
    opnode = g.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(g.add_var("",b))

    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
        outtensor.addtograph(out)
    else:
        outtensor=out
    outtensor.node.add_input(opnode)
    if g.eager:
        ir=DeepxIR(op, [a.node.name,b], [outtensor.node.name],author)
        send(ir)
    return outtensor
def _a_B_elementwiseop_C(
        a: Union[ float, int] ,
        b: Tensor,
        op:str=None,
        out:Union[Tensor,str]="",author='miaobyte')->Tensor:
    g=b.graph
    opnode = g.add_op(op)
    opnode.add_input(g.add_var("",a))
    opnode.add_input(b.node)

    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=b.shape, dtype=b.dtype, device=b.device)
        outtensor.addtograph(out)
    else:
        outtensor=out
    outtensor.node.add_input(opnode)
    if g.eager:
        ir=DeepxIR(op, [a,b.node.name], [outtensor.node.name],author)
        send(ir)
    return outtensor

#add
OpNode.register("add")
class Add(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        return _A_B_elementwiseop_C(a, b, "add", out,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        return out_grad, out_grad
OpNode.register("addscalar")
class AddScalar(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        return _A_b_elementwiseop_C(a, b, "addscalar", out,author)
 
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, None
def add(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor):
        return Add.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        return AddScalar.apply(a,b,out,author,requires_grad=requires_grad)


#sub
OpNode.register("sub")
class Sub(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        return _A_B_elementwiseop_C(a, b, "sub", out,author)
    
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, -grad_output
    
OpNode.register("subscalar")
class SubScalar(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        return _A_b_elementwiseop_C(a, b, "subscalar", out,author)
    
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, None
def sub(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:  
    if isinstance(b,Tensor):
        return Sub.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        return SubScalar.apply(a,b,out,author,requires_grad=requires_grad)

#mul
OpNode.register("mul")
class Mul(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a,b)
        return _A_B_elementwiseop_C(a, b, "mul", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad * b, out_grad * a
    
OpNode.register("mulscalar")
class MulScalar(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _A_b_elementwiseop_C(a, b, "mulscalar", out,author)
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b, None
def mul(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor):
        return Mul.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        return MulScalar.apply(a,b,out,author,requires_grad=requires_grad)
 

#div
OpNode.register("div")
class Div(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a,b)
        return _A_B_elementwiseop_C(a, b, "div", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad / b, -out_grad * a / b / b
    
OpNode.register("divscalar")
class DivScalar(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _A_b_elementwiseop_C(a, b, "divscalar", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad / b, None
    
OpNode.register("rdivscalar")
class RDivScalar(Function):
    @staticmethod
    def forward(ctx:Context, a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _a_B_elementwiseop_C(a, b, "rdivscalar", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b, None
def div(
        a: Optional[Union[Tensor, float, int]] = None,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor) and isinstance(a,Tensor):
        return Div.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        if isinstance(a,Tensor):
            #C=A/b
            return DivScalar.apply(a,b,out,author,requires_grad=requires_grad)
        else:
            #C=a/B
            return RDivScalar.apply(a,b,out,author,requires_grad=requires_grad)

OpNode.register("compare")
class Compare(Function):
    @staticmethod
    def forward(ctx:Context,a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a,b)
        return _A_B_elementwiseop_C(a,b,"compare",out,author)

 
OpNode.register("max")
class Max(Function):
    @staticmethod
    def forward(ctx:Context,a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            mask=_A_B_elementwiseop_C(a,b,"compare",'mask',author)
            ctx.save_tensors(mask)
        return _A_B_elementwiseop_C(a,b,"max",out,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        mask_a=ctx.get_tensor
        mask_b=1-mask_a
        return out_grad*mask_a, out_grad*mask_b
    

OpNode.register("maxscalar")
class MaxScalar(Function):
    @staticmethod
    def forward(ctx:Context,a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _A_b_elementwiseop_C(a,b,"maxscalar",out,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        b=ctx.get_data('b')
        return out_grad, out_grad


def max(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return MaxScalar.apply(a,b,out,author,requires_grad)
    else:
        return Max.apply(a,b,out,author,requires_grad=requires_grad)


OpNode.register("min")
class Min(Function):
    @staticmethod
    def forward(ctx:Context,a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a,b)
        return _A_B_elementwiseop_C(a,b,"min",out,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        a,b=ctx.get_tensors()
        return out_grad, out_grad

OpNode.register("minscalar")
class MinScalar(Function):
    @staticmethod
    def forward(ctx:Context,a,b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _A_b_elementwiseop_C(a,b,"minscalar",out,author)
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        b=ctx.get_data('b')
        return out_grad, out_grad

def min(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return MinScalar.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        return Min.apply(a,b,out,author,requires_grad=requires_grad)

#clamp,TODO

#sqrt
OpNode.register("sqrt")
class Sqrt(Function):
    @staticmethod
    def forward(ctx:Context, a,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a)
        return _A_elementwiseop_C(a,"sqrt",out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad / (2 * sqrt(a)), None
    
def sqrt(
        input:Tensor,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Sqrt.apply(input,out,author,requires_grad=requires_grad)

OpNode.register("pow")
class Pow(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a,b)
        return _A_B_elementwiseop_C(a, b, "pow", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad * b * pow(a,b-1), out_grad * pow(a,b) * log(a)

OpNode.register("powscalar")
class PowScalar(Function):
    @staticmethod
    def forward(ctx:Context, a, b,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_data('b',b)
        return _A_b_elementwiseop_C(a, b, "powscalar", out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b * pow(a,b-1), out_grad * pow(a,b) * log(a)
    
def pow(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return PowScalar.apply(a,b,out,author,requires_grad=requires_grad)
    else:
        return Pow.apply(a,b,out,author,requires_grad=requires_grad)

#exp
OpNode.register("exp")
class Exp(Function):
    @staticmethod
    def forward(ctx:Context, a,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a)
        return _A_elementwiseop_C(a,"exp",out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad * exp(a), None
    
def exp(
        a:Tensor,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Exp.apply(a,out,author,requires_grad=requires_grad)  
#log
OpNode.register("log")
class Log(Function):
    @staticmethod
    def forward(ctx:Context, a,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a)
        return _A_elementwiseop_C(a,"log",out,author)
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad / a, None
    
def log(
        a:Tensor,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Log.apply(a,out,author,requires_grad=requires_grad)

OpNode.register("rsqrt")
class Rsqrt(Function):
    @staticmethod
    def forward(ctx:Context, a,out,author='miaobyte'):
        if ctx.requires_grad:
            ctx.save_tensors(a)
        return _A_elementwiseop_C(a,"rsqrt",out,author)

    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return -out_grad / (2 * a * sqrt(a)), None
    
def rsqrt(
        input:Tensor,
        out:Union[Tensor,str]='',
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Rsqrt.apply(input,out,author,requires_grad=requires_grad)

 


# OpNode.register("Placeholder", 102)
# OpNode.register("Neg", 103)
# NodeType.register("Less", 104)
# NodeType.register("Equal", 105)
 
# NodeType.register("Tanh", 107)
 
 
# def placeholder(name=None, shape=None):
#     node = OpNode("Placeholder", name)
#     if shape:
#         node.set_attr("shape", shape)
#     return node

# def neg(x):
#     node = OpNode("Neg")
#     node.add_input("x", x)
#     return node
 
# def less(a, b):
#     node = OpNode("Less")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def equal(a, b):
#     node = OpNode("Equal")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node
 
# def tanh(x):
#     node = OpNode("Tanh")
#     node.add_input("x", x)
#     return node
 