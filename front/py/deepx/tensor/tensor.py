from typing import Optional 
from .shape import Shape
from .devicetype import Device
from .dtype import infer_dtype,default_dtype

class Tensor:

    #生命周期 
    def __init__(
            self,
            data=None,
            shape=None,
            device=None,
            dtype:Optional[str]=None,
    ):
        # data
        if data is not None:
            import numpy as np
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.data = data           
            self._shape = Shape(data.shape)
        
        # dtype
        if dtype is None:
            if data is not None:
                self._dtype = infer_dtype(data)
            else:
                self._dtype = default_dtype
        else:
            self._dtype = str(dtype)
        
        # format
        if self._dtype == 'float32' or self._dtype == 'float64' or self._dtype == 'float16' or self._dtype == 'bfloat16':
            self._format = '%.4f'
        elif self._dtype == 'int32' or self._dtype == 'int64' or self._dtype == 'int8' or self._dtype == 'int16':
            self._format = '%d'
        else:
            self._format = '%s'
        # shape
        if shape is not None:
            if isinstance(shape, (tuple, list)) and all(isinstance(i, int) for i in shape):
                self._shape = Shape(shape)  # 这里会将列表/元组转换为Shape对象
            elif isinstance(shape, Shape):
                self._shape = shape
            else:
                raise ValueError("Invalid shape")

        # device
        if isinstance(device, str):
            self._device = Device.from_string(device)
        elif isinstance(device, Device):
            self._device = device
        else:
            self._device = Device.CPU  # 默认设备
        self._graph = None
        self._node = None

    # todo，待实现eager模式下的tensor释放
    def __del__(self):
        try:
            if self.graph.eager:
                from deepx.nn.functional import deltensor
                deltensor(self)
        except:
            pass

    # shape
    @property
    def shape(self,dim:int=None):
        if dim is None:
            return self._shape.shape
        else:
            return self._shape.shape[dim]
    @property
    def Shape(self):
        return self._shape
        
    @property
    def stride(self):
        return self._shape.stride
 

    def dim(self):
        return self._shape.dim() if self._shape else None

    @property
    def size(self):
        return self._shape.shape if self._shape else None  

    def size(self,dim:int):
        return self._shape[dim] if self._shape else None  

    @property
    def ndimension(self):
        return self._shape.ndimension() if self._shape else None
    
    @property
    def ndim(self):
        return self._shape.ndim  if self._shape else None
 
    def numel(self)->int:
        return self._shape.numel() if self._shape else None
    
    
    #dtype device
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    
    # 计算图
    def addtograph(self,name:str)->'Tensor':
        # graph
        from deepx.nn.functional import newtensor
        newtensor(self,name)
        return self
    
    @property
    def graph(self):
        return self._graph
     
    @property
    def node(self):
        return self._node
    
    #elementwise
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __truediv__(self, other):
        return self.div(other)
    
    def __rtruediv__(self, other):
        return self.rdiv(other)

    #矩阵乘法
    def __matmul__(self, other):
        return self.matmul(other)

    #shape操作
    @property
    def T(self) -> str:
        return self.transpose(1,0,out=self.node.name+".T")

    # 打印
    def set_format(self,format:str):
        self._format = format
    def __repr__(self) -> str:
        from deepx.nn.functional import printtensor
        s=printtensor(self,format=self._format)
        return s


def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f