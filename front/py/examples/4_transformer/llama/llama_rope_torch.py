hidden_size = 8
eps = 1e-6
dir = '/home/lipeng/model/deepxmodel/llama/'
model_path = "/home/lipeng/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
print()

from transformers import AutoTokenizer, AutoConfig


def init_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


tokenizer = init_tokenizer(model_path)
config = AutoConfig.from_pretrained(model_path)


def tokenize_text(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids
    import torch
    # 处理超出词汇表范围的token
    if torch.any(tokens >= tokenizer.vocab_size):
        # 获取UNK token ID，如果没有则使用0
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer,
                                                         'unk_token_id') and tokenizer.unk_token_id is not None else 0
        # 替换所有超出范围的token为UNK
        tokens = torch.where(tokens < tokenizer.vocab_size, tokens, torch.tensor(unk_token_id, device=tokens.device))
    return tokens


############-------PyTorch-------################
import torch

# 创建输入
text = "这是一个测试文本，用于演示嵌入层的使用。"
torch_input = tokenize_text(text, tokenizer)
from deepxutil.torch import save_torch

save_torch(torch_input, dir + 'input')


# 创建网络

class NetTorch(torch.nn.Module):
    from transformers.models.llama.modeling_llama import LlamaConfig
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        print("rotary_emb.inv_freq")
        print(self.rotary_emb.inv_freq)
    def forward(self, x):
        inputs_embeds = self.embed_tokens(x)
        print(inputs_embeds)
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        return self.rotary_emb(hidden_states, position_ids)

if __name__ == "__main__":
    torch_net = NetTorch(config)
    save_torch(torch_net.embed_tokens.weight, dir + 'weight')
    # 前向传播
    torch_output = torch_net(torch_input)
    torch_sin, torch_cos = torch_output

    print("sin shape:", torch_sin.shape)
    print("sin:", torch_sin)

    print("cos shape:", torch_cos.shape)
    print("cos:", torch_cos)
