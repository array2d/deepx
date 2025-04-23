from deepx.tensor import Tensor
from deepx.nn.functional import newtensor

# 数学公式：relu(x) = max(0, x)
def relu(t: Tensor)->Tensor:
    from .leaffunc_elementwise import max as max_func
    outtensor=t
    if t.name!=None:
        outtensor=newtensor(t.shape, dtype=t.dtype)
    else:#inplace操作
        pass
    return max_func(t,0,outtensor)
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(t: Tensor)->Tensor:
    outtensor=t
    if t.name is not None:
        outtensor=newtensor(t.shape, dtype=t.dtype)
    t.mul(-1,out=outtensor)
    outtensor.exp_()
    outtensor.add_(1)
    outtensor.rdiv_(1)
    return outtensor

# 数学公式：swish(x) = x * σ(βx)
def swish(x: Tensor,beta: float = 1.0) -> Tensor:
    outtensor=x
    if x.name is not None:
        outtensor=newtensor(x.shape, dtype=x.dtype)
    x.mul(beta,out=outtensor)
    outtensor=sigmoid(outtensor)
    outtensor.mul_(x)
    return outtensor
