from deepx.nn.modules import Module
from deepx import  cat,Tensor
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS
from deepx.utils import Config

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(Module):
    def __init__(self,config:Config):
        super().__init__()
        # 最大序列长度
        self.max_seq_len_cached = config.max_position_embeddings
        # 原始最大序列长度
        self.original_max_seq_len = config.max_position_embeddings
        # 旋转类型
        self.rope_type=config.rope_scaling.rope_type
        # 旋转初始化函数
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # 旋转初始化函数
        self.inv_freq, self.attention_scaling = self.rope_init_fn(config)
        self.original_inv_freq = self.inv_freq

    # def _dynamic_frequency_update(self, position_ids, device):
    #     """
    #     dynamic RoPE layers should recompute `inv_freq` in the following situations:
    #     1 - growing beyond the cached sequence length (allow scaling)
    #     2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
    #     """
    #     seq_len = torch.max(position_ids) + 1
    #     if seq_len > self.max_seq_len_cached:  # growth
    #         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
    #         self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
    #         self.max_seq_len_cached = seq_len

    #     if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
    #         # This .to() is needed if the model has been moved to a device after being initialized (because
    #         # the buffer is automatically moved, but not the original copy)
    #         self.original_inv_freq = self.original_inv_freq.to(device)
    #         self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
    #         self.max_seq_len_cached = self.original_max_seq_len

    def forward(self, x, position_ids):
        # 扩展旋转频率
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand((position_ids.shape[0], -1, 1))
 
        # 使用torch.unsqueeze和type转换替代索引操作
        position_ids_expanded = position_ids[:, None, :].float()

        
        # 计算频率
        freqs = (inv_freq_expanded @ position_ids_expanded).mT
        # 拼接频率
        emb = cat((freqs, freqs), dim=-1)
        # 计算余弦和正弦
        cos = emb.cos()
        sin = emb.sin()
        # 应用缩放因子
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.todtype(x.dtype), sin.todtype(x.dtype)

def rotate_half(x:Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return  cat((-x2, x1,), dim=-1)

def apply_rotary_pos_emb(q:Tensor, k:Tensor, cos:Tensor, sin:Tensor, unsqueeze_dim:int=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed