
from .leaffunc_life import *
from .leaffunc_io import *
from .leaffunc_init import *
from .leaffunc_changeshape import *
from .leaffunc_elementwise import *
from .leaffunc_matmul import matmul
from .leaffunc_reduce import reducemax,reducemin,sum,prod

from .authormap import defaultauthor

from .reduce import mean
from .activite import *
from .elementwise import *
from .normalization import *
from .changeshape import *
__all__ = [

    #leaffunc
    "newtensor","rnewtensor","printtensor","load", #life
    "printtensor","save",#io
    "constant","constant_","dropout","full","zeros","ones","uniform","uniform_","arange","arange_",
    "kaiming_uniform","kaiming_uniform_",
    "add","sub","mul","div",
    "sqrt","pow","exp","log",
    "min","max",
    "less","greater","equal","notequal",
    "switch",
    "todtype",
    "invert",
    "matmul",
    "reducemax","reducemin","sum","prod",
    "reshape","permute","transpose","concat","broadcastTo","indexselect",

    #functional
    "relu","sigmoid","swish","silu",
    "mean",
    "rsqrt",
    "softmax",
    "squeeze","unsqueeze",

    #other
    "calculate_fan_in_and_fan_out",
]