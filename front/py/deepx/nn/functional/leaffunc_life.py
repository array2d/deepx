from deepx.tensor import Tensor
from typing import Union

def parse_shape(shape:Union[tuple,list])->tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return tuple(int(dim) for dim in shape)

def newtensor(*shape,dtype:str='float32',name:str=None):
    s=parse_shape(shape)
    t=Tensor(shape=s,dtype=dtype,name=name)
    from .rtf_life import rtf_newtensor
    rtf_newtensor(t)
    return t
def rnewtensor(t:Tensor):
    from .rtf_life import rtf_newtensor
    rtf_newtensor(t)
    return t
def copytensor(t:Tensor,out:Tensor):
    from .rtf_life import rtf_copytensor
    rtf_copytensor(t,out)
def deltensor(t:Tensor):
    from .rtf_life import rtf_deltensor
    rtf_deltensor(t)
def renametensor(t:Tensor,new_name:str):
    assert isinstance(t,Tensor)
    assert isinstance(new_name,str) and new_name != ''
    assert t.name is not None and t.name != ''

    from .rtf_life import rtf_renametensor
    rtf_renametensor(t,new_name)

def load(path:str)->Tensor:
    from .rtf_io import rtf_load
    return rtf_load(path)
