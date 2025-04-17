from .tensor import Tensor,Shape,Number
from deepx.nn.functional import *  # 导入所有functional函数
from deepx.nn.functional import __all__ as _func_all  # 获取functional的导出列表

__all__ = [
    #tensor
    'Tensor','Shape','Number',
    *_func_all
]

# 为了支持 import deepx as dx 的用法
tensor = Tensor