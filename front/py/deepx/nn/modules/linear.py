from .module import Module
from deepx  import Tensor
from deepx.nn.functional import uniform_,kaiming_uniform_,calculate_fan_in_and_fan_out
import math

class Linear(Module):
    r'''
    copy from torch.nn.Linear

    math:`y = xA^T + b` 
    not y=Ax+b
    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype:str="float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(shape=(out_features, in_features),dtype=dtype)
        self.register_parameter("weight",self.weight)
        if bias:
            self.bias = Tensor(shape=(out_features,),dtype=dtype)
            self.register_parameter("bias",self.bias)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        #`y = xA^T + b`
        y=input @ self.weight.T
        oldshape=y.shape
        if self.bias is not None:
            y.reshape_(tuple(y.shape[1:]))
            y=y+self.bias
        y.reshape_(oldshape)
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

