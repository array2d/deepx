hidden_size = 8
eps = 1e-6
dir='/home/lipeng/model/deepxmodel/llama/'



############### PyTorch 实现部分 ###############
import torch
# 使用小规模数据以便打印完整结果
pt_input = torch.arange(48, dtype=torch.float32).reshape(2, 3, hidden_size) / 10.0 - 2.0
print("PyTorch 输入:")
print(pt_input)

from transformers.models.llama.modeling_llama import LlamaRMSNorm as TransformersLlamaRMSNorm
from deepxutil.torch import save_torch
save_torch(pt_input,dir+'rmsnorm_input')
# 使用transformers库中的官方LlamaRMSNorm实现
pt_norm = TransformersLlamaRMSNorm(hidden_size, eps=eps)
# 设置权重为固定值0.5
with torch.no_grad():
    pt_norm.weight.fill_(0.5)
# 前向传播
pt_output = pt_norm(pt_input)


print("\nPyTorch RMSNorm 结果:")
print(pt_output.shape)
print(pt_output)

 
############### DeepX 实现部分 ###############
from deepx import  constant_,load
from deepx.transformer.models.llama.modeling_llama import LlamaRMSNorm

input=load(dir+'rmsnorm_input')

# DeepX计算流程
norm = LlamaRMSNorm(hidden_size=hidden_size, eps=eps)
# 设置相同的权重
constant_(norm.weight, 0.5)
# 前向计算
output = norm(input)
output.print()
