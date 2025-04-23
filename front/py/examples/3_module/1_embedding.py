from transformers import AutoTokenizer
print()
def init_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = init_tokenizer("/home/lipeng/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

def tokenize_text(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids
    import torch
    # 处理超出词汇表范围的token
    if torch.any(tokens >= tokenizer.vocab_size):
        # 获取UNK token ID，如果没有则使用0
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None else 0
        # 替换所有超出范围的token为UNK
        tokens = torch.where(tokens < tokenizer.vocab_size, tokens, torch.tensor(unk_token_id, device=tokens.device))
    return tokens

dir="/home/lipeng/model/deepxmodel/embeddingtest/"
 
############-------PyTorch-------################
import torch.nn as nn

# 创建输入
text = "这是一个测试文本，用于演示嵌入层的使用。"
torch_input = tokenize_text(text, tokenizer)
from deepxutil.torch import save_torch
save_torch(torch_input,dir+'input')
print(torch_input)
# 创建网络
torch_net = nn.Embedding(tokenizer.vocab_size, 4096)
save_torch(torch_net.weight,dir+'weight')
# 前向传播
torch_output = torch_net(torch_input)
print()
print(torch_output.shape)
print(torch_output)


############-------DEEPX-------################
from deepx.nn.modules import Embedding
from deepx.nn.functional import load

input=load(dir+'input')
input.print()

weight=load(dir+'weight')
net = Embedding(tokenizer.vocab_size, 4096,weight=weight)
out=net.forward(input)
out.print()

