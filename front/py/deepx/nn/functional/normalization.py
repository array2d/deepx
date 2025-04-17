
from typing import Union
from deepx import Tensor
from deepx.nn.functional import sub

# 数学公式：softmax(x_i) = e^{x_i} / sum(e^{x_j})
def softmax(t: Tensor,dim:int=-1)->Tensor:

    # 数值稳定性处理：减去最大值防止指数爆炸
    if dim is not None:
        reducemax_t = t.reducemax(dim=[dim], keepdim=True)  # 保持维度用于广播
    else:
        reducemax_t = t.reducemax(keepdim=True)
    t_subed=reducemax_t.broadcastTo(t.shape)
    sub(t,t_subed,out=t_subed)
    # 实现公式：exp(t - max) / sum(exp(t - max))
    exp_t = t_subed.exp()
    expt_reducedsum=exp_t.sum(dim=[dim], keepdim=True)
    expt_sum=expt_reducedsum.broadcastTo(t.shape)
    # 处理输出张量（参考sigmoid的实现模式）
    exp_t.div_(expt_sum)
    return exp_t