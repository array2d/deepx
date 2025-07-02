
from deepx.nn.modules import Module
from deepx import Tensor,ones,rsqrt

# 论文 https://arxiv.org/abs/1910.07467
# 来自 https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
class T5LayerNorm( Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = ones((hidden_size,))
        self.register_parameter("weight", self.weight)
        self.variance_epsilon = eps

    def forward(self, x:Tensor):
        xtype=x.dtype
        # layer norm should always be calculated in float32
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x*rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).todtype(xtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    

RMSNorm = T5LayerNorm