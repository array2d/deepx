from typing import Any
from .optimizer import Optimizer
from deepx.tensor import Tensor

class SGD(Optimizer):
    def __init__(self,
                params:list[Tensor],
                defaults: dict[str, Any]) -> None:
        super().__init__(params, defaults)

    def step(self):
        for param in self.params:
            param.data -= self.defaults['lr'] * param.grad
