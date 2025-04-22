from deepx.tensor import Tensor,Shape,saveShape
from .authormap import defaultauthor

def printtensor(t:Tensor,format=''):
    from .rtf_io import rtf_printtensor
    rtf_printtensor(t,format,defaultauthor['print'])
    return ''

def save(t:Tensor,path:str):
    from .rtf_io import rtf_save
    rtf_save(t,path)
    return t

def save_npy(t,path:str):
    r'''
    保存numpy.ndarray为deepx.tensor格式
    '''
    from numpy import ascontiguousarray
    shape=Shape(t.shape)
    shape._dtype=str(t.dtype)
    saveShape(shape,path+".shape")

    array = ascontiguousarray(t)
    array.tofile(path+'.data')
    return t

def save_torch(t,path:str):
    r'''
    保存torch.Tensor为deepx.tensor格式
    '''
    from torch import Tensor as torch_Tensor
    if isinstance(t,torch_Tensor):
        t=t.detach().cpu().numpy()
    else:
        raise ValueError("t must be a torch.Tensor")
    save_npy(t,path)
    