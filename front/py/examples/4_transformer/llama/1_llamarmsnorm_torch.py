############### PyTorch 实现部分 ###############
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# 使用小规模数据以便打印完整结果
hidden_size = 8
pt_input_data = torch.arange(48, dtype=torch.float32).reshape(2, 3, hidden_size) / 10.0 - 2.0
pt_input = pt_input_data.clone()
eps = 1e-6
print("PyTorch 输入:")
print(pt_input)
# 使用transformers库中的官方LlamaRMSNorm实现
pt_norm = LlamaRMSNorm(hidden_size, eps=eps)
# 设置权重为固定值0.5
with torch.no_grad():
    pt_norm.weight.fill_(0.5)
# 前向传播
pt_output = pt_norm(pt_input)


print("\nPyTorch RMSNorm 结果:")
print(pt_output)
