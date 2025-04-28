from typing import Optional,Union

from deepx.tensor import Tensor,tensor_method,Number

@tensor_method
def add(self,
        other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import add as add_func
    return add_func(self,other,out)

@tensor_method
def add_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import add as add_func
    add_func(self,other,self)
     

@tensor_method
def sub(self, other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import sub as sub_func
    return sub_func(self,other,out)

@tensor_method
def sub_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import sub as sub_func
    sub_func(self,other,self)

@tensor_method
def mul(self,  other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import mul as mul_func
    return mul_func(self,other,out)

@tensor_method
def mul_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import mul as mul_func
    mul_func(self,other,self)

@tensor_method
def div(self, other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import div as div_func
    return div_func(self,other,out)



@tensor_method
def div_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import div as div_func
    div_func(self,other,self)


@tensor_method
def rdiv(self,other:Union[float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import div as div_func
    return div_func(other,self,out)


@tensor_method
def rdiv_(self, other:Union[float,int]):
    from deepx.nn.functional import div as div_func
    div_func(other,self,self)
    return self

@tensor_method
def min(self,  other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import min  as min_func
    return min_func(self,other,out)


@tensor_method
def min_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import min  as min_func
    min_func(self,other,self)
    return self

@tensor_method
def max(self,  other:Union[Tensor,float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import max as max_func
    max_func(self,other,out)
    return out

@tensor_method
def max_(self, other:Union[Tensor,float,int]):
    from deepx.nn.functional import max as max_func
    max_func(self,other,self)
 

@tensor_method
def clamp(self, min:Union[float,int], max:Union[float,int],
        out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import clamp as clamp_func
    return clamp_func(self,min,max,out)
 

@tensor_method
def clamp_(self, min:Union[float,int], max:Union[float,int]):
    #todo
    pass
 

@tensor_method
def exp(self,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import exp as exp_func
    return exp_func(self,out)
 

@tensor_method
def exp_(self):
    from deepx.nn.functional import exp as exp_func
    exp_func(self,self)
 

@tensor_method
def pow(self,
        b:Union[float,int],
        out:Union[Tensor,str]=''):
    from deepx.nn.functional import pow as pow_func
    return pow_func(self,b,out)
 

@tensor_method
def pow_(self,
        b:Union[float,int]):
    from deepx.nn.functional import pow as pow_func
    pow_func(self,b,self)

@tensor_method
def rpow(self,
        a:Number,
        out:Union[Tensor,str]=''):
    from deepx.nn.functional import rpow as rpow_func
    return rpow_func(a,self,out)


@tensor_method
def sqrt(self,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import sqrt as sqrt_func
    return sqrt_func(self,out)
 
@tensor_method
def sqrt_(self):
    from deepx.nn.functional import sqrt as sqrt_func   
    sqrt_func(self,self)

@tensor_method
def rsqrt(self,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import rsqrt as rsqrt_func
    return rsqrt_func(self,out)

@tensor_method
def rsqrt_(self):
    from deepx.nn.functional import rsqrt as rsqrt_func
    rsqrt_func(self,self)

@tensor_method
def invert(self,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import invert as invert_func
    return invert_func(self,out)


@tensor_method
def dropout_(self,p:float=0.5,seed:int=None):
    from deepx.nn.functional import dropout as dropout_func
    dropout_func(self,p,seed)
    return self

