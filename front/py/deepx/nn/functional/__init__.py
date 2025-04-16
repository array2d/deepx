
from .leaffunc_new import newtensor,deltensor
from .leaffunc_io import printtensor
from .leaffunc_init import *
from .leaffunc_changeshape import *
from .leaffunc_elementwise import *
from .leaffunc_matmul import matmul
from .leaffunc_reduce import reducemax,reducemin,sum,prod

from .reduce import mean

from .activite import relu,sigmoid,swish
__all__ = [
    #leaffunc
    "newtensor",
    "printtensor",
    "constant","constant_","full","zeros","ones","uniform","uniform_","arange","arange_","kaiming_uniform","kaiming_uniform_","calculate_fan_in_and_fan_out",
    "add","sub","mul","div","sqrt","pow","exp","log","rsqrt",
    "leaffunc_matmul",
    "reducemax","reducemin","sum","prod",
    "reshape","permute","transpose","concat","broadcastTo",
    "relu","sigmoid","swish",

    #func
    "mean",
]