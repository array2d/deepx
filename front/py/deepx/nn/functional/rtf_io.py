from deepx.tensor import Tensor
from deepx.nn import DeepxIR,Param
from deepx.scheduler import send

def rtf_printtensor(t:Tensor,format='',author='miaobyte'):
    args=[Param.tensor(t),Param.varstr(format)]
    returns=[]
    ir=DeepxIR("print", args, returns,author)
    send(ir)
    return ''

def rtf_load(t:Tensor,path:str,author='miaobyte'):
    args=[Param.tensor(t),Param.varstr(path)]
    returns=[]
    ir=DeepxIR("load", args, returns,author)
    send(ir)
    return t

def rtf_save(t:Tensor,path:str,author='miaobyte'):
    args=[Param.tensor(t),Param.varstr(path)]
    returns=[]
    ir=DeepxIR("save", args, returns,author)
    send(ir)
    return t