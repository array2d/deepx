from deepx.tensor import Tensor
from deepx.nn.functional import newtensor
from .leaffunc_elementwise  import exp
# 数学公式：relu(x) = max(0, x)
def relu(t: Tensor)->Tensor:
    from .leaffunc_elementwise import max as max_func
    outtensor=newtensor(t.shape, dtype=t.dtype)
    return max_func(t,0,outtensor)
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(t: Tensor)->Tensor:
    return 1 / (exp(t*-1)+1)

# 数学公式：swish(x) = x * σ(βx)
def swish(x: Tensor,beta: float = 1.0) -> Tensor:
    outtensor=x
    if x.name is not None:
        outtensor=newtensor(x.shape, dtype=x.dtype)
    x.mul(beta,out=outtensor)
    outtensor=sigmoid(outtensor)
    outtensor.mul_(x)
    return outtensor
