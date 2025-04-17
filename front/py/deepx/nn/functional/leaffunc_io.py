from deepx.tensor import Tensor
from .authormap import defaultauthor

def printtensor(t:Tensor,format=''):
    from .rtf_io import rtf_printtensor
    rtf_printtensor(t,format,defaultauthor['print'])
    return ''

