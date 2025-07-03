from deepx.tensor import Shape
from numpy import ascontiguousarray,ndarray

def save_numpy(t,tensorpath:str,realdtype:str=None):
    r'''
    保存numpy.ndarray为deepx.tensor格式
    t:numpy.ndarray
    tensorpath:str,
    '''

    assert isinstance(t,ndarray)
    shape=Shape(t.shape)
    shape._dtype=str(t.dtype)
    if realdtype is not None:
        shape._realdtype=realdtype
    shape.save(tensorpath+".shape")

    array = ascontiguousarray(t)
    array.tofile(tensorpath+'.data')
    return t
