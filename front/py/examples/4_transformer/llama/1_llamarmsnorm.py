
# 使用小规模数据以便打印完整结果
hidden_size = 8
eps = 1e-6


############### DeepX 实现部分 ###############
from deepx import arange, constant_
from deepx.transformer.models.llama.modeling_llama import LlamaRMSNorm

# 使用相同的数据
input = arange(2, 3, hidden_size, dtype="float32")
input.div_(10.0)
input.sub_(2.0)
eps = 1e-6

input.print()

# DeepX计算流程
norm = LlamaRMSNorm(hidden_size=hidden_size, eps=eps)
# 设置相同的权重
constant_(norm.weight, 0.5)
# 前向计算
output = norm(input)
output.print()
