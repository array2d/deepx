############-------PyTorch-------################
import torch
from token_text import  torch_input,config

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
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        return self.rotary_emb(hidden_states, position_ids)

if __name__ == "__main__":
    torch_net = NetTorch(config)
    # 前向传播
    torch_output = torch_net(torch_input)
    torch_sin, torch_cos = torch_output

    print("sin shape:", torch_sin.shape)
    print("sin:", torch_sin)

    print("cos shape:", torch_cos.shape)
    print("cos:", torch_cos)
