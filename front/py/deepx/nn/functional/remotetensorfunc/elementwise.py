from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def add(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("add", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def add_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("addscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def sub(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("sub", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def sub_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("subscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def mul(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("mul", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def mul_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("mulscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def div(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("div", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def div_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("divscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def rdiv_scalar(a:float, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("rdivscalar", [a, b.node.name], [out.node.name], author)
    send(ir)
    return out


def sqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("sqrt", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def pow(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("pow", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def pow_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("powscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def exp(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("exp", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def log(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("log", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def rsqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("rsqrt", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def sin(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("sin", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def cos(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("cos", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def tan(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("tan", [a.node.name], [out.node.name], author)
    send(ir)
    return out

def compare(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("compare", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def max(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("max", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def max_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("maxscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out

def min(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("min", [a.node.name, b.node.name], [out.node.name], author)
    send(ir)
    return out

def min_scalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    ir=DeepxIR("minscalar", [a.node.name, b], [out.node.name], author)
    send(ir)
    return out