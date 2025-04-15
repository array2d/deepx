from typing import Optional,Union
from deepx import Tensor

def relu(
        t: Tensor,
        inplace:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    from .leaffunc_elementwise import max as max_func
    return max_func(t,0,out)
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(
        t: Tensor,
        inplace:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    """Sigmoid激活函数

    .. math::
        \sigma(x) = \frac{1}{1 + e^{-x}}

    Args:
        t: 输入张量
        inplace: 是否原地操作
        out: 输出张量或名称

    Returns:
        输出张量
    """
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    from .leaffunc_elementwise import exp
    outtensor = 1 / ((t*-1).exp()+1)
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
