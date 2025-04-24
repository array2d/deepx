
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
__all__ = [

    #leaffunc
    "newtensor","rnewtensor","printtensor","load", #life
    "printtensor","save",#io
    "constant","constant_","full","zeros","ones","uniform","uniform_","arange","arange_","kaiming_uniform","kaiming_uniform_","calculate_fan_in_and_fan_out",
    "add","sub","mul","div","sqrt","pow","exp","log",
    "matmul",
    "reducemax","reducemin","sum","prod",
    "reshape","permute","transpose","concat","broadcastTo","indexselect",

    #functional
    "relu","sigmoid","swish",
    "mean",
    "rsqrt",
    "softmax",

]