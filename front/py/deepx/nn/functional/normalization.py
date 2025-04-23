from deepx import Tensor

# 数学公式：softmax(x_i) = e^{x_i} / sum(e^{x_j})
def softmax(t: Tensor,dim:int=-1)->Tensor:

    # 数值稳定性处理：减去最大值防止指数爆炸
    if dim is not None:
        reducemax_t = t.reducemax(dim=[dim], keepdim=True)  # 保持维度用于广播
    else:
        reducemax_t = t.reducemax(keepdim=True)
    t_subed=t.clone()
    t_subed.sub_(reducemax_t)

    # 实现公式：exp(t_subed) / sum(exp(t_subed))
    exp_t = t_subed.exp()
    expt_sum=exp_t.sum(dim=[dim], keepdim=True)
    # 处理输出张量（参考sigmoid的实现模式）
    exp_t.div(expt_sum,out=t_subed)
    return t_subed