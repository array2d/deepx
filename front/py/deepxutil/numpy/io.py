from deepx.tensor import Shape
from numpy import ascontiguousarray,ndarray

def save_numpy(t,tensorpath:str):
    r'''
    保存numpy.ndarray为deepx.tensor格式
    t:numpy.ndarray
    tensorpath:str,
    '''

    assert isinstance(t,ndarray)
    shape=Shape(t.shape)
    shape._dtype=str(t.dtype)
    shape.save(tensorpath+".shape")

    array = ascontiguousarray(t)
    array.tofile(tensorpath+'.data')
    return t
