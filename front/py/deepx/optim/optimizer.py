from typing import Any
from deepx.tensor import Tensor
class Optimizer:
    def __init__(self, 
                 params:list[Tensor],
                 defaults: dict[str, Any]) -> None:
        self.params = params
        self.defaults = defaults

    def step(self):
        pass