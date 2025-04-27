from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send

def rtf_newtensor(t:Tensor):
    assert isinstance(t,Tensor)
    args=[Param.vector(t.shape,'int32')]
    returns=[Param.tensor(t)]
    ir=DeepxIR("newtensor", args, returns,'')
    send(ir)


def rtf_copytensor(t:Tensor,out:Tensor):
    assert isinstance(t,Tensor)
    assert isinstance(out,Tensor)
    assert t.shape==out.shape
    assert t.dtype==out.dtype

    args=[Param.tensor(t)]
    returns=[Param.tensor(out)]
    ir=DeepxIR("copytensor", args, returns,'')
    send(ir)



def rtf_deltensor(t:Tensor):
    assert isinstance(t,Tensor)
    args=[]
    returns=[Param.tensor(t)]
    ir=DeepxIR("deltensor", args, returns,'')
    send(ir)

def rtf_renametensor(t:Tensor,new_name:str):
    args=[Param.varstr(new_name)]
    returns=[Param.tensor(t)]
    ir=DeepxIR("renametensor", args, returns,'')
    send(ir)
