from deepx.tensor import Shape,saveShape
 
def save_numpy(t,tensorpath:str):
    r'''
    保存numpy.ndarray为deepx.tensor格式
    t:numpy.ndarray
    tensorpath:str,
    '''
    from numpy import ascontiguousarray,ndarray
    assert isinstance(t,ndarray)
    shape=Shape(t.shape)
    shape._dtype=str(t.dtype)
    saveShape(shape,tensorpath+".shape")

    array = ascontiguousarray(t)
    array.tofile(tensorpath+'.data')
    return t
