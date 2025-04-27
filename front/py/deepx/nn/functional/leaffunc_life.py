from deepx.tensor import Tensor
from typing import Union
 
def newtensor(shape:tuple[int,...],dtype:str='float32',name:str=None):
    assert isinstance(shape,tuple)
    for i in shape:
        assert isinstance(i,int)
    assert isinstance(dtype,str)
    assert isinstance(name,str) or name is None

    t=Tensor(shape=shape,dtype=dtype,name=name)
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
