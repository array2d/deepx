from token_text import torch_input
print()
############-------TORCH-------################
import torch
from  transformers.models.llama.modeling_llama import rotate_half,LlamaAttention

model_path = "/home/lipeng/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
from deepx.utils import Config
config=Config.from_file(model_path+"config.json")
config._attn_implementation= "eager"

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

class NetTorch(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.get("pad_token_id", None)
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.attn = LlamaAttention(config, layer_idx=0)

    def forward(self, x):
        # 1. 词嵌入
        inputs_embeds = self.embed_tokens(x)
        hidden_states = inputs_embeds
        # 2. 位置编码
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        # 3. RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        # 4. Attention
        attn_output, attn_weights = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None
        )
        return attn_output, attn_weights

if __name__ == "__main__":
    torch_net = NetTorch(config)
    attn_output, attn_weights = torch_net(torch_input)
    print("attn_output shape:", attn_output.shape)
    print("attn_output:", attn_output)
    print("attn_weights shape:", attn_weights.shape)

