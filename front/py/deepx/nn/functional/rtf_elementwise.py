from deepx.tensor import Tensor,Number
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send
from .rtf import A_B_op_C,A_scalar_op_C,A_op_C


def rtf_add(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("add",a,b,out,author)
    return out

def rtf_addscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("addscalar",a,b,out,author)
    return out

def rtf_sub(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("sub",a,b,out,author)
    return out

def rtf_subscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("subscalar",a,b,out,author)
    return out

def rtf_mul(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("mul",a,b,out,author)
    return out

def rtf_mulscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("mulscalar",a,b,out,author)
    return out

def rtf_div(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("div",a,b,out,author)
    return out

def rtf_divscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("divscalar",a,b,out,author)
    return out

def rtf_rdivscalar(a:float, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    args = [ Param.varnum(a),Param.tensor(b)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("rdivscalar", args, returns, author)
    send(ir)
    return out

def rtf_sqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("sqrt",a,out,author)
    return out

def rtf_pow(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("pow",a,b,out,author)
    return out

def rtf_powscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("powscalar",a,b,out,author)
    return out

def rtf_rpowscalar(a:Number,b:Tensor,out:Tensor,author='miaobyte')->Tensor:
    args = [ Param.varnum(a),Param.tensor(b)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("rpowscalar", args, returns, author)
    send(ir)
    return out

def rtf_exp(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("exp",a,out,author)
    return out

def rtf_log(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("log",a,out,author)
    return out

def rtf_rsqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("rsqrt",a,out,author)
    return out

def rtf_sin(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("sin",a,out,author)
    return out

def rtf_cos(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("cos",a,out,author)
    return out

def rtf_tan(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("tan",a,out,author)
    return out

def rtf_compare(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("compare",a,b,out,author)
    return out

def rtf_max(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("max",a,b,out,author)
    return out

def rtf_maxscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("maxscalar",a,b,out,author)
    return out

def rtf_min(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("min",a,b,out,author)
    return out

def rtf_minscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("minscalar",a,b,out,author)
    return out

def rtf_invert(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("invert",a,out,author)
    return out

def rtf_todtype(t:Tensor,dest:Tensor):
    assert isinstance(t,Tensor)
    assert isinstance(dest,Tensor)
    assert t.shape==dest.shape

    args=[Param.tensor(t)]
    returns=[Param.tensor(dest)]
    ir=DeepxIR("todtype", args, returns,'')
    send(ir)

def rtf_dropout(a:Tensor, p:float, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("dropout",a,p,out,author)
    return out