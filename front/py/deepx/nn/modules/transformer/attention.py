from typing import Optional,Tuple
from deepx import Tensor,matmul,softmax,dropout

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    scaling_factor: float = 1.0,
    dropout_prob: float = 0.0
) -> Tuple[Tensor, Tensor]:
   
    # 参考论文: https://arxiv.org/abs/1706.03762 (Attention is All You Need)
    #1 计算注意力分数
    attn_scores = (query @ key.mT) * scaling_factor

    #2 应用注意力掩码
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_scores = attn_scores + causal_mask


    #3 softmax归一化
    attn_weights =  softmax(attn_scores, dim=-1)


    #4 可选的dropout
    if dropout_prob > 0.0:
        attn_weights = dropout(attn_weights, p=dropout_prob)
    
    #5 注意力加权值
    attn_output =  matmul(attn_weights, value)

    return attn_output, attn_weights
