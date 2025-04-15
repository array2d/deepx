from deepx.tensor import Tensor

def printtensor(t:Tensor,format='',author='miaobyte'):
    from .rtf_io import rtf_printtensor
    rtf_printtensor(t,format,author)
    return ''

