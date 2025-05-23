from typing import Union
from deepx.tensor import tensor_method

# 填充
@tensor_method
def full_(self,value:Union[float,int]):
    from deepx.nn.functional import constant_ as constant_func
    constant_func(self,value=value)

@tensor_method
def dropout_(self,p:float=0.5,seed:int=None):
    from deepx.nn.functional import dropout as dropout_func
    dropout_func(self,p,seed)
    return self


@tensor_method
def zeros_(self):
    from deepx.nn.functional import constant_ as constant_func
    constant_func(self,value=0)

@tensor_method
def ones_(self):
    from deepx.nn.functional import constant_ as constant_func
    constant_func(self,value=1)

@tensor_method
def uniform_(self,low=0, high=1,seed:int=None):
    from deepx.nn.functional import uniform_ as uniform_func
    uniform_func(self,low=low, high=high,seed=seed)

@tensor_method
def arange_(self,start=0,step=1):
    from deepx.nn.functional import arange_ as arange_func
    arange_func(self,start,step)

@tensor_method
def normal_(self,mean=0, stddev=1,seed:int=None):
    from deepx.nn.functional import normal_ as normal_func
    normal_func(self,mean,stddev,seed)

@tensor_method
def rand_(self):
    #todo
    pass

@tensor_method
def randn_(self):
    #todo
    pass
@tensor_method
def eye_(self,n,m=None):
    #todo
    pass
