import yaml
import os
from deepx.tensor import Shape,Tensor,tensor_method

def loadShape(path:str)->tuple[str,Shape,str]:
    filename = os.path.basename(path)
    if filename.endswith('.shape'):
        with open(path, 'r') as f:
            shape = yaml.safe_load(f)
    else:
        raise ValueError("文件名必须以.shape结尾")
 
    tensor_name = filename[:-6]  # 移除'.shape'后缀
    return (tensor_name,Shape(tuple(shape['shape'])),shape['dtype'])
@tensor_method
def loadData(self,path:str):
    from deepx.nn.functional import loadData as loadData_func
    loadData_func(self,path)
    
@tensor_method
def save(self,path:str):
    from deepx.nn.functional import save  as save_func
    save_func(self,path)

 
