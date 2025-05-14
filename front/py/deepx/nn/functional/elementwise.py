from deepx.tensor import Tensor
from deepx.nn.functional import newtensor

# 幂运算
def rsqrt(input:Tensor)->Tensor:
    from .leaffunc_elementwise import sqrt
    return 1/sqrt(input)
 
# 比较
def clamp(input:Tensor,min:float,max:float)->Tensor:
    from .leaffunc_elementwise import max,min
    return max(min(input,max),min)

# 类型转换
def double(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='float64',name=input.name)
    return todtype(input,dest)

def float(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='float32',name=input.name)
    return todtype(input,dest)

def float16(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='float16',name=input.name)
    return todtype(input,dest)
def bfloat16(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='bfloat16',name=input.name)
    return todtype(input,dest)

def int64(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='int64',name=input.name)
    return todtype(input,dest)
def int32(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='int32',name=input.name)
    return todtype(input,dest)

def int16(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='int16',name=input.name)
    return todtype(input,dest)

def int8(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='int8',name=input.name)
    return todtype(input,dest)

def bool(input:Tensor)->Tensor:
    from .leaffunc_elementwise import todtype
    dest=newtensor(input.shape,dtype='bool',name=input.name)
    return todtype(input,dest)

def where(condition:Tensor,x:Tensor,y:Tensor)->Tensor:
    from .leaffunc_elementwise import switch as switch_func
    return switch_func((y,x),condition)