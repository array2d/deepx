from typing import Optional,Union,TypeAlias
from triton.language.semantic import equal
from .shape import Shape


Number: TypeAlias = Union[int, float, bool]

 
class Tensor:

    #life
    def __init__(self,shape:tuple[int,...],dtype:str='float32',name:str=None):
        # name
        assert isinstance(name,str) or name is None
        assert isinstance(shape,tuple)
        for i in shape:
            assert isinstance(i,int) and i>0
        assert isinstance(dtype,str)

        self._name = name
        if name is None or name =='':
            if not hasattr(self.__class__, '_instance_counter'):
                self.__class__._instance_counter = 0
            count = self.__class__._instance_counter
            self.__class__._instance_counter += 1
            self._name = str(count)
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
        assert isinstance(t,Tensor)
        assert t.name != self._name
        from deepx.nn.functional import copytensor
        copytensor(self,t)

    def clone(self,name:str=None):
        from deepx.nn.functional import copytensor,newtensor
        t=newtensor(self.shape,dtype=self.dtype,name=name)
        copytensor(self,t)
        return t
    def to(self,dtype:str,name:str=None):
        assert isinstance(dtype,str) and dtype != ''
        from deepx.nn.functional import todtype as todtype_func,newtensor
        dest=newtensor(self.shape,dtype=dtype,name=name)
        todtype_func(self,dest)
        return dest
    # name
    @property
    def name(self):
        return self._name
    
    def rtf_rename(self,name:str):
        assert isinstance(name,str) and name != ''
        assert self.name is not None and self.name != ''

        from deepx.nn.functional import renametensor
        renametensor(self,name)
        self._name = name
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
    def __radd__(self, other:Union[Number,'Tensor']):
        return self.add(other)
    def __sub__(self, other:Union[Number,'Tensor']):
        return self.sub(other)
    def __rsub__(self, other:Union[Number,'Tensor']):
        return self.rsub(other)
    def __mul__(self, other:Union[Number,'Tensor']):
        return self.mul(other)
    def __rmul__(self, other:Union[Number,'Tensor']):
        return self.mul(other)
    def __neg__(self):
        return self.mul(-1.0)
    def __truediv__(self, other:Union[Number,'Tensor']):
        return self.div(other)
    def __rtruediv__(self, other:Union[Number,'Tensor']):
        return self.rdiv(other)
    # 幂指
    def __pow__(self, other:Union[Number,'Tensor']):
        return self.pow(other)
    
    def __rpow__(self, other:Union[Number,'Tensor']):
        return self.rpow(other)
    # 位
    def __invert__(self):
        return self.invert()
    # 比较
    def __eq__(self, other:Union[Number,'Tensor']):
        return self.equal(other)
    def __ne__(self, other:Union[Number,'Tensor']):
        return self.notequal(other)
    def __gt__(self, other:Union[Number,'Tensor']):
        return self.greater(other)
    def __ge__(self, other:Union[Number,'Tensor']):
        return other.less(self)
    def __lt__(self, other:Union[Number,'Tensor']):
        return self.less(other)
    def __le__(self, other:Union[Number,'Tensor']):
        return other.greater(self)   
    
    #矩阵乘法
    def __matmul__(self, other:'Tensor'):
        return self.matmul(other)
    def __rmatmul__(self, other:'Tensor'):
        return other.matmul(self)

    def __getitem__(self, idx):
        # 简单操作
        if  isinstance(idx,Tensor):
            return self.indexselect(idx)
        if isinstance(idx, int): 
            return self.sliceselect(slice(idx,idx+1)).squeeze(dim=0)
        
        ## 阶段1,
        if isinstance(idx, slice):
            indices = [idx]
        elif isinstance(idx, tuple):
            indices = list(idx)
        else:
            raise TypeError(f"Index must be an integer, slice, tuple, or Tensor, not {type(idx).__name__}")
        # 阶段2
        result = self
        new_axis_positions = []
        dim_cursor = 0
        
        for item in  indices:
            if item is None:
                # 如果是 None，则表示在该位置添加一个新的维度
                new_axis_positions.append(dim_cursor)
                continue
            if item == Ellipsis:
                num_ellipsis = self.ndim - len(indices) + 1
                dim_cursor += num_ellipsis
                continue
            # 如果是完整的切片 (e.g., ':')，则无需操作，直接进入下一维度
            if item == slice(None, None, None):
                dim_cursor += 1
                continue
            result=result.sliceselect(item,dim=dim_cursor)
            dim_cursor += 1
 
        # 2. 在指定位置添加新维度（由 None 产生）
        i=0
        for pos in sorted(new_axis_positions):
            result = result.unsqueeze(pos+i)
            i += 1

        return result

    #shape操作
    @property
    def mT(self) -> str:
        return self.transpose(-1,-2)

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