from deepx.tensor import Tensor
from .authormap import defaultauthor

def printtensor(t:Tensor,format=''):
    from .rtf_io import rtf_printtensor
    rtf_printtensor(t,format,defaultauthor['print'])
    return ''

def save(t:Tensor,path:str):
    from .rtf_io import rtf_save
    rtf_save(t,path)
    return t
 
def loadData(t:Tensor,path:str)->Tensor:
    from .rtf_io import rtf_loadtensordata
    return rtf_loadtensordata(t,path)