from .tensor import *
from .shape import Shape
from .elementwise import *  # 导入所有包含@tensor_method装饰的方法
from .matmul import *       # 导入矩阵乘法相关方法
from .changeshape import *    # 导入转置方法
from .init import *
from .reduce import *
from .io import *
__all__ = [
    'Shape',
    'Tensor',
    'tensor_method',
    'Number',
    'loadShape',
    # 'lt', 'gt', 'eq',
    # 'sin', 'cos', 'tan',
    # 'DType',
    # '_dtype_to_typestr'
] 