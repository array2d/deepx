from typing import Optional,Union
import math

from deepx import Tensor
from deepx.autograd import OpNode,Function,Context
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

OpNode.register("constant")
class Constant(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                value:Optional[Union[float,int]]=None,
                author='miaobyte') -> Tensor:
        opnode = t.graph.add_op("constant")
        argnode=t.graph.add_var('',value)   
        opnode.add_input(argnode)
        t.node.add_input(opnode)
        if t.graph.eager:
            ir=DeepxIR("constant",  [Param(t.node.name, 'tensor', t.dtype),Param(value)], [],author)
            send(ir)
        return t
def constant(t:Tensor,
            value:Optional[Union[float,int]]=None,
            author='miaobyte')->Tensor:
    return Constant.apply(t,value,author)

def full(*shape, value=0, dtype=None, device=None,
         name:Union[Tensor,str]='')->Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    outtensor=None
    if isinstance(name,str):
        outtensor=Tensor(shape=shape, dtype=dtype, device=device)
        outtensor.addtograph(name)
    else:
        outtensor=name
    return constant(outtensor, value)

def zeros(*shape, dtype=None, device=None,
         name:Union[str]='')->Tensor:
    return full(*shape, value=0, dtype=dtype, device=device,name=name)

def ones(*size, dtype=None, device=None,
         name:Union[str]='')->Tensor:
    return full(*size, value=1, dtype=dtype, device=device,name=name)

OpNode.register("arange")
class Arange(Function):
    @staticmethod
    def forward(ctx:Context,
                start:Optional[Union[float,int]]=0,
                end:Optional[Union[float,int]]=None,
                step:Optional[Union[float,int]]=1,dtype=None, device=None,name:Union[Tensor,str]='',author='miaobyte')->Tensor:
        outtensor=None
        if isinstance(name,str):
            shape=[end-start]
            outtensor=Tensor(shape=shape, dtype=dtype, device=device)
            outtensor.addtograph(name)
        else:
            outtensor=name
        g=outtensor.graph
        if g.eager:
            ir=DeepxIR("arange",  [outtensor.node.name,start,step], [],author)
            send(ir)
        return outtensor
def arange(start=0, end=None, step=1,dtype=None, device=None,name:Union[Tensor,str]='',author='miaobyte')->Tensor:
    return Arange.apply(start,end,step,dtype,device,name,author)

OpNode.register("uniform")
class Uniform(Function):
    @staticmethod
    def forward(ctx:Context,
                t:Tensor,
                low:Optional[Union[float,int]]=0,
                high:Optional[Union[float,int]]=1,
                seed:Optional[int]=0,author='miaobyte')->Tensor:
        if low >= high:
                raise ValueError(f"low({low})必须小于high({high})")
        if t is None:
            raise ValueError("t不能为None")
        g=t.graph
    
        opnode = g.add_op("uniform")
        opnode.add_input(g.add_var('',low))
        opnode.add_input(g.add_var('',high))
        if seed is not None:
            opnode.add_input(g.add_var('',seed))
        t.node.add_input(opnode)
        if t.graph.eager:
            ir=DeepxIR("uniform",  [t.node.name,low, high,seed], [],author)
            send(ir)
        return t


def uniform(t:Tensor,low=0, high=1,seed:int=0,author='miaobyte')->Tensor:
    return Uniform.apply(t,low,high,seed,author)

def rand(*size, dtype=None, device=None):
   #TODO
   pass

def randn(*size, dtype=None, device=None):
    #TODO
    pass

def eye(
        n:int,
        m:Optional[int]=None,
        dtype:Optional[str]=None, 
        device:Optional[str]=None):
    #TODO
    pass
 

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
    return tensor.uniform_(-bound, bound)
