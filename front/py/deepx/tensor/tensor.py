import uuid
from typing import Optional,Union
from .shape import Shape


class Tensor:

    #生命周期 
    def __init__(self,shape:Union[tuple[int],list[int],Shape],dtype:str='float32',name:str=None):
        # name

        self._name = name
        if name =='':
            self._name =None
        # dtype
        self._dtype = dtype
        
        # format
        self.autoformat()
        # shape
 
        if isinstance(shape, (tuple, list)) and all(isinstance(i, int) for i in shape):
            self._shape = Shape(shape)  # 这里会将列表/元组转换为Shape对象
        elif isinstance(shape, Shape):
            self._shape = shape
        else:
            raise ValueError("Invalid shape")
 
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
    
    # name
    @property
    def name(self):
        return self._name
    
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
    
    
    #dtype 
    @property
    def dtype(self):
        return self._dtype
 
    
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
    def autoformat(self):
        if self._dtype == 'float32' or self._dtype == 'float64' or self._dtype == 'float16' or self._dtype == 'bfloat16':
            self._format = '%.4f'
        elif self._dtype == 'int32' or self._dtype == 'int64' or self._dtype == 'int8' or self._dtype == 'int16':
            self._format = '%d'
        else:
            self._format = '%s'
    def set_format(self,format:str):
        self._format = format
    def __repr__(self) -> str:
        from deepx.nn.functional import printtensor
        s=printtensor(self,format=self._format)
        return s


def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f