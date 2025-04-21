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
    保存numpy.tensor为deepxtensor格式
    '''
    from numpy import save,ndarray,ascontiguousarray
    shape=Shape(t.shape)
    shape._dtype=str(t.dtype)
    saveShape(shape,path+".shape")

    array = ascontiguousarray(t)
    array.tofile(path+'.data')
    return t