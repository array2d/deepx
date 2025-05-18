from typing import   Tuple
import math
from deepx import arange,Tensor,where

def _compute_default_rope_parameters(config:dict={
    "rope_theta":10000.0,
    "head_dim":0,
    "partial_rotary_factor":1.0,
}) -> Tuple[Tensor, float]:
    partial_rotary_factor = config.get("partial_rotary_factor", 1.0)
    dim   = config["head_dim"]* partial_rotary_factor
    # 计算逆频率
    base=config["rope_theta"]
    inv_freq = 1.0 / (base ** (arange(0, dim, 2, dtype='float64')/ dim))
    return inv_freq, 1.0
    
def _compute_llama3_parameters(config:dict={
    "rope_theta":10000.0,
    "head_dim":0,
    "partial_rotary_factor":1.0,
    "factor":8,
    "low_freq_factor":1,
    "high_freq_factor":4,
    "old_context_len":8192,
    "seq_len":None
}) -> Tuple[Tensor, float]:
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config)

    factor = config["rope_scaling"]["factor"]  # `8` in the original implementation
    low_freq_factor = config["rope_scaling"]["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config["rope_scaling"]["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config["rope_scaling"]["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    wavelen.print()
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    is_medium_freq.print()
    # TODO 这一步执行后，会导致an illegal memory access was encountered
    inv_freq_llama =  where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    is_medium_freq.print()
    inv_freq_llama.print()
    return inv_freq_llama, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    # "linear": _compute_linear_scaling_rope_parameters,
    # "dynamic": _compute_dynamic_ntk_parameters,
    # "yarn": _compute_yarn_parameters,
    # "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}
  