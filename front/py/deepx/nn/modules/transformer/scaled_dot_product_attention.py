from typing import Optional,Tuple
from deepx import Tensor,matmul,softmax,dropout

def scaled_dot_product(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    scaling_factor: float = 1.0,
    dropout_prob: float = 0.0
) -> Tuple[Tensor, Tensor]:
    # 计算注意力分数
    attn_scores = (query @ key.mT) * scaling_factor

    # softmax归一化
    attn_weights =  softmax(attn_scores, dim=-1)

    # 应用注意力掩码
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # 可选的dropout
    if dropout_prob > 0.0:
        attn_weights = dropout(attn_weights, p=dropout_prob)
    
    # 注意力加权值
    attn_output =  matmul(attn_weights, value)
    
    # 恢复原始维度
    attn_output = attn_output.mT
    return attn_output, attn_weights
