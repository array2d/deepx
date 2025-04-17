from deepx.tensor import Tensor
from typing import Optional,Union
from .leaffunc_reduce import sum

#mean
 
def mean(
        a:Tensor,
        dim:Optional[Union[list[int],tuple[int]]]=None,
        keepdim:bool=False,
        out:Union[str]='')->Tensor:
    # 如果dim为None,则对所有维度求平均
    if dim is None:
        dim = list(range(a.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    total = 1
    for i in dim:
        if i < 0:
            dim[i] = i + a.dim()
        total *= a.shape[i]
    result = sum(a, dim, keepdim, out)/total
    return result
