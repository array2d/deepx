from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def rtf_newtensor(t:Tensor,name:str=None):
    args=[Param.vector(t.shape,'int32')]
    returns=[Param.tensorName(name,t.dtype)]
    ir=DeepxIR("newtensor", args, returns,'')
    send(ir)


def rtf_copytensor(t:Tensor,out:Tensor):
    args=[Param.tensor(t)]
    returns=[Param.tensor(out)]
    ir=DeepxIR("copytensor", args, returns,'')
    send(ir)

def rtf_deltensor(t:Tensor):
    args=[Param.tensor(t)]
    returns=[]
    ir=DeepxIR("deltensor", args, returns,'')
    send(ir)
