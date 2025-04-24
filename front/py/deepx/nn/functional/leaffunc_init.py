import math
import time
import os
from .leaffunc_life import newtensor
from .rtf_init import *
from deepx import Tensor,Number
from .authormap import defaultauthor

# 命名规则
# inplace操作的函数，其名为_后缀, 返回值为空
# 非inplace操作的函数，其名为_后缀, 返回值为Tensor

def constant_(t:Tensor,value: Union[float,int])->Tensor:
    rtf_constant(t,value,defaultauthor['constant'])
    
def constant(shape:tuple[int,...], value:Union[float,int], dtype:str='float32',name:str=None)->Tensor:
    outtensor=newtensor(shape,dtype=dtype,name=name)
    constant_(outtensor, value)
    return outtensor

def full(shape:tuple[int,...], value:Union[float,int], dtype:str='float32',name:str=None)->Tensor:
    return constant(shape, value=value, dtype=dtype,name=name)

def zeros(shape:tuple[int,...], dtype:str='float32',name:str=None)->Tensor:
    return constant(shape, value=0, dtype=dtype,name=name)

def ones(shape:tuple[int,...], dtype:str='float32',name:str=None)->Tensor:
    return constant(shape, value=1, dtype=dtype,name=name)
 
def arange_(t:Tensor,start=0,step=1)->Tensor:
    from .rtf_init import rtf_arange
    rtf_arange(t,start,step,defaultauthor['arange'])
#pytorch style
def arange(start:Number,end:Number,step:Number=1,dtype:str='float32',name:str=None)->Tensor:
    s =[int((end-start)/step)]
    outtensor=newtensor(s,dtype=dtype,name=name)
    arange_(outtensor,start,step)
    return outtensor

def uniform_(t:Tensor,low=0, high=1,seed:int=None)->Tensor:
    if seed is None:
        seed = int(time.time() * 1000) & 0xffffffff
        seed = (seed + os.getpid()) & 0xffffffff
    from .rtf_init import rtf_uniform
    rtf_uniform(t,low,high,seed,defaultauthor['uniform'])

def uniform(shape:tuple[int,...],low=0, high=1,seed:int=None,dtype:str='float32',name:str=None)->Tensor:
    outtensor=newtensor(shape,dtype=dtype,name=name)
    uniform_(outtensor,low,high,seed)
    return outtensor

def calculate_fan_in_and_fan_out(tensor:Tensor)->tuple[int,int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor:Tensor, mode:str)->tuple[int,int]:
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out

#copy from torch.nn/init.py
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
)->Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return  uniform_(tensor,-bound, bound)

def kaiming_uniform(shape:tuple[int,...],a:float=0,mode:str='fan_in',nonlinearity:str='leaky_relu',dtype:str='float32',name:str=None,author='miaobyte')->Tensor:
    outtensor=newtensor(shape,dtype=dtype,name=name)
    kaiming_uniform_(outtensor,a,mode,nonlinearity)
    return outtensor

def normal_(t:Tensor,mean:float=0, stddev:float=1,seed:int=None)->Tensor:
    if seed is None:
        seed = int(time.time() * 1000) & 0xffffffff
        seed = (seed + os.getpid()) & 0xffffffff
    from .rtf_init import rtf_normal
    rtf_normal(t,mean,stddev,seed,defaultauthor['normal'])

def normal(shape:tuple[int,...],mean:float=0, stddev:float=1,seed:int=None,dtype:str='float32',name:str=None,author='miaobyte')->Tensor:
    outtensor=newtensor(shape,dtype=dtype,name=name)
    normal_(outtensor,mean,stddev,seed)
    return outtensor
