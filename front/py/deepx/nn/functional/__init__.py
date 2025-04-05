from .elementwise import *
from .new import newtensor,deltensor
from .print import printtensor
from .matmul import matmul
from .init import *
from .reduce import reduce_max,reduce_min,sum,prod,mean
from .changeshape import transpose,reshape,broadcast_shape,broadcast_to,unsqueeze
from .activite import relu,sigmoid,swish
__all__ = [
    "newtensor",
    "printtensor",
    "constant","full","zeros","ones","uniform","arange","rand","randn","eye","kaiming_uniform_","calculate_fan_in_and_fan_out",
    "add","sub","mul","div","sqrt","pow","exp","log","rsqrt",
    "matmul",
    "max","min","sum","prod","mean",
    "transpose","reshape","broadcast_shape","broadcast_to","unsqueeze",
    "relu","sigmoid","swish",
    
]