
from .leaffunc_new import newtensor,deltensor
from .leaffunc_io import printtensor
from .leaffunc_init import *

from .leaffunc_changeshape import *
from .elementwise import *
from .matmul import matmul

from .reduce import reduce_max,reduce_min,sum,prod,mean

from .activite import relu,sigmoid,swish
__all__ = [
    "newtensor",
    "printtensor",
    "constant","constant_","full","zeros","ones","uniform","uniform_","arange","arange_","kaiming_uniform","kaiming_uniform_","calculate_fan_in_and_fan_out",
    "add","sub","mul","div","sqrt","pow","exp","log","rsqrt",
    "matmul",
    "max","min","sum","prod","mean",
    "reshape","permute","transpose","concat","broadcast_to",
    "relu","sigmoid","swish",
    
]