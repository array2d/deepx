from typing import Optional
from deepx import Tensor, Module
from .attention import  scaled_dot_product_attention

def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# 经简化，去掉了分布式配置，去掉attention的配置。交给IR自动替换flashattention，后续的组件自动处理


def grouped_query_attention(
    module: Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling_factor: float,
    dropout_prob: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    return scaled_dot_product_attention(
        query, key, value,
        attention_mask=attention_mask,
        scaling_factor=scaling_factor,
        dropout_prob=dropout_prob
    )
