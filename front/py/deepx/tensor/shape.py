import numpy as np
from typing import Optional,Union
class Shape:
    def __init__(self, shape:tuple[int,...]=None):
        # 确保 shape 是元组类型
        assert isinstance(shape,tuple)
        self._shape = shape
        for i in self._shape:
            assert isinstance(i,int) and i>0
        self._size = int(np.prod(self.shape)) if self.shape else 0
        # 计算 stride（步长）
        self._strides = self._compute_strides()
        self._dtype=None
        
    @property
    def shape(self,dim=None):
        if dim is None:
            return self._shape
        else:
            return self._shape[dim]
        
    def numel(self)->int:
        """计算张量中所有元素的数量（与torch.Tensor.numel()行为一致）
        
        实现说明：
        - 使用np.prod计算所有维度的乘积
        - 空shape时返回0（对应标量情况）
        - 返回int类型保持与PyTorch一致
        """
        return self._size  # 在__init__中已预先计算好

    def dim(self)->int:
        """返回张量的维度数（与torch.Tensor.dim()行为一致）
        
        实现说明：
        - 直接返回_shape元组的长度
        - 处理空shape的情况（对应标量返回0）
        - 与PyTorch的dim()返回int类型保持一致
        """
        return len(self._shape)

    @property
    def ndim(self)->int:
        """返回张量的维度数（dim的别名，与PyTorch命名习惯保持一致）
        
        设计考虑：
        - 保持与PyTorch的ndimension()别名一致性
        - 实际调用dim()方法避免代码重复
        - 使用更符合Python风格的命名方式
        """
        return self.dim()
    
    def ndimension(self)->int:
        """返回张量的维度数（dim的别名，与PyTorch命名习惯保持一致）
        
        设计考虑：
        - 保持与PyTorch的ndimension()别名一致性
        - 实际调用dim()方法避免代码重复
        - 使用更符合Python风格的命名方式
        """
        return self.dim()
  
    @property
    def stride(self)->tuple[int,...]:
        """返回所有维度的步长元组"""
        return self._strides

    def _compute_strides(self):
        """计算每个维度的步长"""
        if not self.shape:
            return ()
        strides = [1]
        for dim in reversed(self.shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))
    
    def __str__(self):
        return f"Size({list(self.shape)})" 
    
    def __repr__(self):
        return f"Size({self.shape})"
    
    def __getitem__(self, idx):
        return self.shape[idx]
    
    def __len__(self)->int:
        return len(self.shape)
    
    def __iter__(self):
        return iter(self.shape)
        
    def __eq__(self, other)->bool:
        """比较两个形状是否相等"""
        if isinstance(other, Shape):
            return self.shape == other.shape
        elif isinstance(other, (tuple, list)):
            return self.shape == tuple(other)
        return False
        
    def __hash__(self):
        """使Shape可哈希，便于在字典和集合中使用"""
        return hash(self.shape)

    @classmethod
    def total_size(cls,other:tuple[int,...])->int:
        total_size=1
        for i in other:
            total_size*=i
        return total_size
    

    @classmethod
    def transpose(cls,shape:tuple[int,...],dimorder:tuple[int,...]=None)->tuple[int,...]:
        if dimorder is None:
            dimorder=tuple(range(len(shape)))
        return Shape(tuple(shape[i] for i in dimorder))
    
    @classmethod
    def matmul(cls,shape:tuple[int],other:tuple[int])->tuple[int]:
        if len(shape)<2 or len(other)<2:
            raise ValueError(f"matmul: self.ndimension()<2 or other.ndimension()<2")
        if len(shape)!=len(other):
            raise ValueError(f"matmul: self.ndimension()!=other.ndimension()")
        if shape[-1]!=other[-2]:
            raise ValueError(f"matmul: self.shape[-1]!=other.shape[-2]")
        resultshape=list(shape)
        resultshape[-1]=other[-1]
        return tuple(resultshape)
    
    @classmethod
    def broadcast_shape(cls,shape_a: tuple[int,...], shape_b: tuple[int,...]) -> tuple[int,...]:
        """计算两个形状的广播后形状"""
        # 获取形状的长度
        len_a, len_b = len(shape_a), len(shape_b)
        
        # 创建结果形状
        result_shape = []
        
        # 从右往左对齐并计算每个维度
        for i in range(1, min(len_a, len_b) + 1):
            dim_a = shape_a[-i]
            dim_b = shape_b[-i]
            
            if dim_a == 1 or dim_b == 1:
                # 广播规则：如果一个维度为1，取另一个维度的值
                result_shape.insert(0, max(dim_a, dim_b))
            elif dim_a == dim_b:
                # 维度相同，保持不变
                result_shape.insert(0, dim_a)
            else:
                # 维度不同且都不为1，无法广播
                raise ValueError(f"无法广播的形状：{shape_a} 和 {shape_b}")
        
        # 添加较长形状中多出的前导维度
        if len_a > len_b:
            result_shape = list(shape_a[:len_a - len_b]) + result_shape
        elif len_b > len_a:
            result_shape = list(shape_b[:len_b - len_a]) + result_shape
        
        return tuple(result_shape)

 
    @classmethod
    def reduceshape(cls,shape:tuple[int,...],dim:tuple[int,...],keepdim:bool)->tuple[int,...]:
        ndim = len(shape)
        # 处理负数维度
        normalized_dim = [d % ndim for d in dim]
        # 去重并排序
        unique_dim = sorted(set(normalized_dim))
        
        if keepdim:
            return tuple(1 if i in unique_dim else s 
                        for i, s in enumerate(shape))
        else:
            return tuple(s for i, s in enumerate(shape)
                        if i not in unique_dim)
    
    # 参考自 https://www.tensorflow.org/api_docs/python/tf/gather
    @classmethod
    def indexselectshape(cls,input_shape:tuple[int,...],index_shape:tuple[int,...],gatheraxis:int)->tuple[int,...]:
        return input_shape[:gatheraxis]+index_shape+input_shape[gatheraxis+1:]

    def save(self,path:str):
        if path.endswith('.shape'):
            import yaml
            with open(path, 'w') as f:
                yaml.dump({'shape': list(self.shape), 'dtype': self._dtype,'size':self.numel(),'dim':self.ndim,'stride':list(self.stride)}, f)
        else:
            raise ValueError("文件名必须以.shape结尾")