from typing import Optional,Union,TypeAlias
from .shape import Shape


Number: TypeAlias = Union[int, float, bool]

tensorid=1

class Tensor:

    #life
    def __init__(self,shape:Union[tuple[int],list[int],Shape],dtype:str='float32',name:str=None):
        # name

        self._name = name
        if name is None or name =='':
            global tensorid
            self._name =tensorid
            tensorid+=1
        # dtype
        self._dtype = dtype

        # shape
 
        if isinstance(shape, (tuple, list)) and all(isinstance(i, int) for i in shape):
            self._shape = Shape(shape)  # 这里会将列表/元组转换为Shape对象
        elif isinstance(shape, Shape):
            self._shape = shape
        else:
            raise ValueError("Invalid shape")

    def copy_to(self,t:'Tensor'):
        from deepx.nn.functional import copytensor
        copytensor(self,t)

    def clone(self,name:str=None):
        from deepx.nn.functional import copytensor,newtensor
        t=newtensor(self.shape,dtype=self.dtype,name=name)
        copytensor(self,t)
        return t
    
    # name
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,name:str):
        self._name=name

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

    
    #elementwise
    def __add__(self, other:Union[Number,'Tensor']):
        return self.add(other)
    
    def __sub__(self, other:Union[Number,'Tensor']):
        return self.sub(other)
    
    def __mul__(self, other:Union[Number,'Tensor']):
        return self.mul(other)
    
    def __truediv__(self, other:Union[Number,'Tensor']):
        return self.div(other)
    
    def __rtruediv__(self, other:Union[Number,'Tensor']):
        return self.rdiv(other)

    def __pow__(self, other:Union[Number,'Tensor']):
        return self.pow(other)
    
    def __rpow__(self, other:Union[Number,'Tensor']):
        return self.rpow(other)
    
    def __invert__(self):
        return self.invert()
    #矩阵乘法
    def __matmul__(self, other:Union[Number,'Tensor']):
        return self.matmul(other)

    #gather
    def __getitem__(self, indices:'Tensor'):
        return self.gather(indices)

    #shape操作
    @property
    def T(self) -> str:
        return self.transpose()

    # 打印
    @staticmethod
    def autoformat(dtype):
        if dtype == 'float32' or dtype == 'float64' or dtype == 'float16' or dtype == 'bfloat16':
            return '%.4f'
        elif dtype == 'int32' or dtype == 'int64' or dtype == 'int8' or dtype == 'int16':
            return '%d'
        elif dtype == 'bool':
            return '%d'
        else:
            return '%s'
 
    def print(self,format:str=None):
        if format is None:
            format=self.autoformat(self.dtype)
        from deepx.nn.functional import printtensor
        printtensor(self,format)
    def __repr__(self) -> str:
        return 'Tensor(shape={},dtype={},name={})'.format(self.shape,self.dtype,self.name)

def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f