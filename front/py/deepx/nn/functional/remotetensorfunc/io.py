from deepx.tensor import Tensor
from deepx.nn import DeepxIR
from deepx.scheduler import send

def printtensor(t:Tensor,format='',author='miaobyte'):
    ir=DeepxIR("print",[t.node.name,format], [],author)
    send(ir)
    return ''

def load(t:Tensor,path:str,author='miaobyte'):
    ir=DeepxIR("load",[t.node.name,path], [],author)
    send(ir)
    return t

def save(t:Tensor,path:str,author='miaobyte'):
    ir=DeepxIR("save",[t.node.name,path], [],author)
    send(ir)
    return t