from typing import Optional, Union
from deepx import Tensor
from deepx.autograd import  Function,Context

from .leaffunc_new import newtensor


class Add(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'add':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_add
        rtf_add(a,b,outtensor,ctx.authormap['add'])
        return outtensor
    @staticmethod
    def backward(ctx:Context,out_grad):
        return out_grad, out_grad
 
class AddScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'addscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_addscalar
        rtf_addscalar(a,b,outtensor,ctx.authormap['addscalar'])
        return outtensor
 
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, None
def add(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor):
        return Add.apply(a,b,out,{'add':author},requires_grad=requires_grad)
    else:
        return AddScalar.apply(a,b,out,{'addscalar':author},requires_grad=requires_grad)


#sub
 
class Sub(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'sub':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_sub
        rtf_sub(a,b,outtensor,ctx.authormap['sub'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, -grad_output
 
class SubScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'subscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_subscalar
        rtf_subscalar(a,b,outtensor,ctx.authormap['subscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, grad_output):
        return grad_output, None
def sub(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:  
    if isinstance(b,Tensor):
        return Sub.apply(a,b,out,{'sub':author},requires_grad=requires_grad)
    else:
        return SubScalar.apply(a,b,out,{'subscalar':author},requires_grad=requires_grad)

#mul
 
class Mul(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'mul':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_mul
        rtf_mul(a,b,outtensor,ctx.authormap['mul'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad * b, out_grad * a
 
class MulScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'mulscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_mulscalar
        rtf_mulscalar(a,b,outtensor,ctx.authormap['mulscalar'])
        return outtensor
        
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b, None
def mul(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor):
        return Mul.apply(a,b,out,{'mul':author},requires_grad=requires_grad)
    else:
        return MulScalar.apply(a,b,out,{'mulscalar':author},requires_grad=requires_grad)
 

#div
 
class Div(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'div':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_div
        rtf_div(a,b,outtensor,ctx.authormap['div'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad / b, -out_grad * a / b / b
 
class DivScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'divscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_divscalar
        rtf_divscalar(a,b,outtensor,ctx.authormap['divscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad / b, None
 
class RDivScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:float,b:Tensor,out:Union[Tensor,str],authormap:dict={'rdivscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(b.shape,dtype=b.dtype,name=out)
        from .rtf_elementwise import rtf_rdivscalar
        rtf_rdivscalar(a,b,outtensor,ctx.authormap['rdivscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b, None
def div(
        a: Optional[Union[Tensor, float, int]] = None,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if isinstance(b,Tensor) and isinstance(a,Tensor):
        return Div.apply(a,b,out,{'div':author},requires_grad=requires_grad)
    else:
        if isinstance(a,Tensor):
            #C=A/b
            return DivScalar.apply(a,b,out,{'divscalar':author},requires_grad=requires_grad)
        else:
            #C=a/B
            return RDivScalar.apply(a,b,out,{'rdivscalar':author},requires_grad=requires_grad)
 


class Max(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'max':'miaobyte'})->Tensor :
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_max
        rtf_max(a,b,outtensor,ctx.authormap['max'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        mask_a=ctx.get_tensor
        mask_b=1-mask_a
        return out_grad*mask_a, out_grad*mask_b
 
class MaxScalar(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'maxscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_maxscalar
        rtf_maxscalar(a,b,outtensor,ctx.authormap['maxscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        b=ctx.get_data('b')
        return out_grad, out_grad


def max(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return MaxScalar.apply(a,b,out,{'maxscalar':author},requires_grad=requires_grad)
    else:
        return Max.apply(a,b,out,{'max':author},requires_grad=requires_grad)

 
class Min(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'min':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_min
        rtf_min(a,b,outtensor,ctx.authormap['min'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        a,b=ctx.get_tensor
        return out_grad, out_grad
 
class MinScalar(Function):
    @staticmethod
    def forward(ctx:Context,a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'minscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_minscalar
        rtf_minscalar(a,b,outtensor,ctx.authormap['minscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad):
        b=ctx.get_data('b')
        return out_grad, out_grad

def min(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return MinScalar.apply(a,b,out,{'minscalar':author},requires_grad=requires_grad)
    else:
        return Min.apply(a,b,out,{'min':author},requires_grad=requires_grad)
 
#sqrt
 
class Sqrt(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor,out:Union[Tensor,str],authormap:dict={'sqrt':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_sqrt
        rtf_sqrt(a,outtensor,ctx.authormap['sqrt'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad / (2 * sqrt(a)), None
    
def sqrt(
        input:Tensor,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Sqrt.apply(input,out,{'sqrt':author},requires_grad=requires_grad)

 
class Pow(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:Tensor,out:Union[Tensor,str],authormap:dict={'pow':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_pow
        rtf_pow(a,b,outtensor,ctx.authormap['pow'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a,b=ctx.get_tensor
        return out_grad * b * pow(a,b-1), out_grad * pow(a,b) * log(a)
 
class PowScalar(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor, b:float,out:Union[Tensor,str],authormap:dict={'powscalar':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_powscalar
        rtf_powscalar(a,b,outtensor,ctx.authormap['powscalar'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        b=ctx.get_data('b')
        return out_grad * b * pow(a,b-1), out_grad * pow(a,b) * log(a)
    
def pow(
        a:Tensor,
        b:Union[int,float,Tensor,]=0,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    if  isinstance(b,int) or isinstance(b,float):
        return PowScalar.apply(a,b,out,{'powscalar':author},requires_grad=requires_grad)
    else:
        return Pow.apply(a,b,out,{'pow':author},requires_grad=requires_grad)

#exp
 
class Exp(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor,out:Union[Tensor,str],authormap:dict={'exp':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_exp
        rtf_exp(a,outtensor,ctx.authormap['exp'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad * exp(a), None
    
def exp(
        a:Tensor,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Exp.apply(a,out,{'exp':author},requires_grad=requires_grad)  
#log

class Log(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor,out:Union[Tensor,str],authormap:dict={'log':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_log
        rtf_log(a,outtensor,ctx.authormap['log'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return out_grad / a, None
    
def log(
        a:Tensor,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Log.apply(a,out,{'log':author},requires_grad=requires_grad)
 
class Rsqrt(Function):
    @staticmethod
    def forward(ctx:Context, a:Tensor,out:Union[Tensor,str],authormap:dict={'rsqrt':'miaobyte'})->Tensor:
        ctx.set_authormap(authormap)
        outtensor=out
        if isinstance(out,str):
            outtensor=newtensor(a.shape,dtype=a.dtype,name=out)
        from .rtf_elementwise import rtf_rsqrt
        rtf_rsqrt(a,outtensor,ctx.authormap['rsqrt'])
        return outtensor

    @staticmethod
    def backward(ctx:Context, out_grad):
        a=ctx.get_tensor
        return -out_grad / (2 * a * sqrt(a)), None
    
def rsqrt(
        input:Tensor,
        out:Union[Tensor,str]=None,
        requires_grad:bool=False,
        author='miaobyte')->Tensor:
    return Rsqrt.apply(input,out,{'rsqrt':author},requires_grad=requires_grad)

  