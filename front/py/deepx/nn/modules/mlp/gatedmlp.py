from deepx.nn.modules import Module,Linear
from .actfn import ACT2FN

class GatedMLP(Module):
    def __init__(self, config:dict):
        super().__init__()
        # 输入层大小
        self.hidden_size = config.hidden_size  
        # 中间层大小
        self.intermediate_size = config.intermediate_size
        #门控投影层
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        #上投影层
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        #下投影层
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        #激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.act_fn(gate)
        out = act * up
        out = self.down_proj(out)
        return out