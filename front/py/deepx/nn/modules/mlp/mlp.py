from deepx.nn.modules import Module, Linear
from .actfn import ACT2FN

class StandardMLP(Module):
    def __init__(self, config: dict):
        super().__init__()
        # 输入层大小
        self.hidden_size = config.hidden_size
        # 中间层大小
        self.intermediate_size = config.intermediate_size
        # 第一层线性
        self.fc1 = Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # 第二层线性
        self.fc2 = Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x