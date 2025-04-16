from deepx.tensor import Tensor
from typing import Optional,Union
from .leaffunc_reduce import sum

#mean
 
def mean(
        a:Tensor,
        dims:Optional[Union[list[int],tuple[int]]]=None,
        keepdim:bool=False,
        out:Union[str]='')->Tensor:
    # 如果dim为None,则对所有维度求平均
    if dims is None:
        dims = list(range(a.ndim))
    elif isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)
    total = 1
    for i in dims:
        if i < 0:
            dims[i] = i + a.dim()
        total *= a.shape[i]
    result = sum(a, dims, keepdim, out)/total
    return result
