from deepx.tensor import Tensor
from deepx.autograd import OpNode
from deepx.nn import DeepxIR
from deepx.scheduler import send

 
def printtensor(t:Tensor,format='',author='miaobyte'):
    from .rtf_io import rtf_printtensor
    rtf_printtensor(t,format,author)
    return ''

