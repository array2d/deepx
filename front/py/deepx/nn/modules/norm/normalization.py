from deepx.nn.modules import Module
from deepx import Tensor,ones,rsqrt
# RMSNorm
# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# 数学公式
class LlamaRMSNorm(Module):
    def __init__(self, hidden_size:int, eps:float=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight=ones((hidden_size,))
        self.register_parameter("weight",self.weight)
        self.variance_epsilon = eps
    def forward(self, hidden_states:Tensor):
        variance =  hidden_states.pow(2).mean((-1,), keepdim=True)
        hidden_states = hidden_states * rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
 
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    