from deepx.tensor import tensor_method

@tensor_method
def full_(self,value):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,value=value)

@tensor_method
def zeros_(self):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,value=0)

@tensor_method
def ones_(self):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,value=1)

@tensor_method
def uniform_(self,low=0, high=1,seed:int=0):
    from deepx.nn.functional import uniform as uniform_func
    uniform_func(self,low=low, high=high,seed=seed)

@tensor_method
def rand_(self):
    #todo
    pass

@tensor_method
def randn_(self):
    #todo
    pass

@tensor_method
def arange_(self,start=0,step=1,author='miaobyte'):
    from deepx.nn.functional import arange_ as arange_func
    arange_func(self,start,step,author)

@tensor_method
def eye_(self,n,m=None):
    #todo
    pass
