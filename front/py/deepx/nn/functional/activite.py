from typing import Union
from deepx.tensor import Tensor
from deepx.nn.functional import newtensor

def relu(t: Tensor,out:Union[Tensor,str]='',requires_grad:bool=False)->Tensor:
    from .leaffunc_elementwise import max as max_func
    return max_func(t,0,out,requires_grad)
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(
        t: Tensor,
        out:Union[Tensor,str]='',
        requires_grad:bool=False)->Tensor:
    outtensor=out
    if isinstance(out,str):
        outtensor=newtensor(t.shape, dtype=t.dtype,name=out)
    
    if requires_grad:
        outtensor = 1 / ((t*-1).exp()+1)
    else:
        t.mul(-1,out=outtensor)
        outtensor.exp_()
        outtensor.add_(1)
        outtensor.rdiv_(1)
    return outtensor

def swish(
        x: Tensor,
        beta: float = 1.0,
        out: Union[Tensor,str] = '') -> Tensor:
    """Swish激活函数
    .. math::
        \text{swish}(x) = x \cdot \sigma(\beta x)
    其中 :math:`\sigma(x)` 是sigmoid函数。
    Args:
        x: 输入张量
        beta: 缩放因子,控制sigmoid的陡峭程度
        out: 输出张量或名称

    Returns:
        输出张量
    """
    return x*sigmoid(x*beta,out=out)
