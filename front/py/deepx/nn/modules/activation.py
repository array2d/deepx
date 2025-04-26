from typing import Union
from deepx  import Tensor,ones
from .module import Module


class Glu(Module):
    def __init__(self):
        super().__init__()
        self.W = ones(shape=(1,1),name=self.full_name+"_W")
        self.V = ones(shape=(1,1),name=self.full_name+"_V")

class Swiglu(Module):
    def __init__(self):
        super().__init__()
        self.W = ones(shape=(1,1),name=self.full_name+"_W")
        self.V = ones(shape=(1,1),name=self.full_name+"_V")

    def swiglu(
        x: Tensor,
        W: Tensor,  # 第一个投影矩阵
        V: Tensor,  # 第二个投影矩阵
        beta: float = 1.0,  # swish函数的缩放因子
        out: Union[Tensor,str] = '') -> Tensor:
        from deepx.nn.functional import swish
        result=swish(x@W,beta=beta).mul(x@V,out=out)       
        return result
 
    def forward(self, input: Tensor) -> Tensor:
        return self.swiglu(input,self.W,self.V)
 
