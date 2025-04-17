
from typing import Union

from deepx.tensor import Tensor,tensor_method

@tensor_method
def reduce_max(self, dims:list[int],keepdim:bool=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_max as reduce_max_func
    return reduce_max_func(self,dims,keepdim,out)

@tensor_method
def reduce_min(self, dims:list[int],keepdim:bool=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_min as reduce_min_func
    return reduce_min_func(self,dims,keepdim,out)


@tensor_method
def sum(self, dims:list[int],keepdim:bool=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import  sum as sum_func
    return  sum_func(self,dims,keepdim,out)

@tensor_method
def prod(self, dims:list[int],keepdim:bool=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import prod as prod_func
    return prod_func(self,dims,keepdim,out)   

@tensor_method
def mean(self, dims:list[int],keepdim:bool=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import mean as mean_func
    return mean_func(self,dims,keepdim,out)
 