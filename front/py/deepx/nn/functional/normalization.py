from deepx import Tensor

# 数学公式：softmax(x_i) = e^{x_i} / sum(e^{x_j})
def softmax(t: Tensor,dim:list[int]=[-1])->Tensor:
    assert isinstance(dim,list)
    for i in range(len(dim)):
        dim[i]=dim[i]%t.ndim
    # 数值稳定性处理：减去最大值防止指数爆炸
    if dim is not None:
        t_reducemax = t.reducemax(dim=tuple(dim), keepdim=True)  # 保持维度用于广播
    else:
        t_reducemax= t.reducemax(keepdim=True)

    t=t-t_reducemax

    t_exp = t.exp()
    t_exp_sum=t_exp.sum(dim=tuple(dim), keepdim=True)
    return t.exp()/t_exp_sum