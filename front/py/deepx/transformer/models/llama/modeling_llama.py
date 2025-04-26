from deepx.nn.modules import Module,Linear
from deepx import Tensor,ones,rsqrt,concat
from deepx.transformer.modeling_rope_utils import ROPE_INIT_FUNCTIONS
# RMSNorm
# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# 数学公式
class LlamaRMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight=ones(hidden_size)
        self.register_parameter("weight",self.weight)
        self.variance_epsilon = eps
    def forward(self, hidden_states:Tensor):
        variance =  hidden_states.pow(2).mean((-1,), keepdim=True)
        hidden_states = hidden_states * rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
 
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    

class LlamaRotaryEmbedding(Module):
    def __init__(self,rope_type:str="default",max_seq_len:int=1024,device=None):
        super().__init__()
        # 最大序列长度
        self.max_seq_len_cached = max_seq_len
        # 原始最大序列长度
        self.original_max_seq_len = max_seq_len
        # 旋转类型
        self.rope_type=rope_type
        # 旋转初始化函数
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # 旋转初始化函数
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        #TODO 
        # 注册缓存
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 原始旋转频率
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
        # TODO
        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
 
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = concat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj