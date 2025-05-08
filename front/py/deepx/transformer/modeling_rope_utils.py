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
    #TODO

    low_freq_wavelen = config.old_context_len / config.low_freq_factor
    high_freq_wavelen = config.old_context_len / config.high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama =  where(wavelen > low_freq_wavelen, inv_freq / config.factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (config.old_context_len / wavelen - config.low_freq_factor) / (config.high_freq_factor - config.low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / config.factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama =  where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor
 
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    # "linear": _compute_linear_scaling_rope_parameters,
    # "dynamic": _compute_dynamic_ntk_parameters,
    # "yarn": _compute_yarn_parameters,
    # "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}
  