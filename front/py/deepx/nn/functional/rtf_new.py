from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def rtf_newtensor(t:Tensor):
    args=[Param.vector(t.shape,'int32')]
    returns=[Param.tensor(t)]
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
