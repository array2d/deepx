from deepx.tensor import Tensor,loadShape
from deepx.nn import DeepxIR,Param
from deepx.scheduler import send

def rtf_printtensor(t:Tensor,format='',author='miaobyte'):
    args=[Param.tensor(t),Param.varstr(format)]
    returns=[]
    ir=DeepxIR("print", args, returns,author)
    send(ir)
    return ''

def rtf_save(t:Tensor,path:str):
    args=[Param.tensor(t),Param.varstr(path)]
    returns=[]
    ir=DeepxIR("save", args, returns)
    send(ir)
    return t

def rtf_load(path:str)->Tensor:
    args=[Param.varstr(path)]
    returns=[]
    ir=DeepxIR("load", args, returns)
    send(ir)
    shapefile=path+'.shape'
    tensor_name,shape,dtype=loadShape(shapefile)
    return Tensor(shape.shape,dtype,tensor_name)
