from deepx.tensor import Tensor
from deepx.autograd.graph import Graph
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def newtensor(t:Tensor,name:str=None):
    from .rtf_new import rtf_newtensor
    rtf_newtensor(t,name)
def copytensor(t:Tensor,out:Tensor):
    from .rtf_new import rtf_copytensor
    rtf_copytensor(t,out)
def deltensor(t:Tensor):
    from .rtf_new import rtf_deltensor
    rtf_deltensor(t)

