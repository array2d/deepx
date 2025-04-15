from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send
from typing import Union
from .rtf import A_B_op_C,A_scalar_op_C,A_op_C

def rtf_add(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("add",a,b,out,author)
    return out

def rtf_add_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("add",a,b,out,author)
    return out

def rtf_sub(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("sub",a,b,out,author)
    return out

def rtf_sub_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("sub",a,b,out,author)
    return out

def rtf_mul(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("mul",a,b,out,author)
    return out

def rtf_mul_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("mul",a,b,out,author)
    return out

def rtf_div(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("div",a,b,out,author)
    return out

def rtf_div_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("div",a,b,out,author)
    return out

def rtf_rdiv_scalar(a:float, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("rdiv",a,b,out,author)
    return out



def rtf_sqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("sqrt",a,out,author)
    return out

def rtf_pow(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("pow",a,b,out,author)
    return out

def rtf_pow_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("pow",a,b,out,author)
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

def rtf_max_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("max",a,b,out,author)
    return out

def rtf_min(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("min",a,b,out,author)
    return out

def rtf_min_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("min",a,b,out,author)
    return out