import yaml
import os
from deepx.tensor import Shape

def loadShape(path:str)->tuple[str,Shape,str]:
    filename = os.path.basename(path)
    if filename.endswith('.shape'):
        with open(path, 'r') as f:
            shape = yaml.safe_load(f)
    else:
        raise ValueError("文件名必须以.shape结尾")
 
    tensor_name = filename[:-6]  # 移除'.shape'后缀
    return (tensor_name,Shape(tuple(shape['shape'])),shape['dtype'])

def saveShape(t:Shape,path:str):
    if path.endswith('.shape'):
        with open(path, 'w') as f:
            yaml.dump({'shape': list(t.shape), 'dtype': t._dtype,'size':t.numel(),'dim':t.ndim,'stride':list(t.stride)}, f)
    else:
        raise ValueError("文件名必须以.shape结尾")

