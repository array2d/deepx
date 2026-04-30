import torch
import numpy as np
import yaml
import os
import sys
def load_shape(path):
    with open(path + '.shape', 'r') as f:
        shape_data = f.read()
    shape = yaml.safe_load(shape_data)
    return shape['shape'], shape['dim'], shape['strides'], shape['size']

def load_tensor_data(path, shape):
    data = np.fromfile(path + '.data', dtype=np.float32)
    return data.reshape(shape)

def load_deepx_tensor(path):
    shape, dim, strides, size = load_shape(path)
    tensor_data = load_tensor_data(path, shape)
    return torch.tensor(tensor_data)

# 使用示例
if __name__ == "__main__":
    name=sys.argv[1]
    tensor = load_deepx_tensor(name)
    print("Tensor:", tensor)
    print("Shape:", tensor.size()) 
    print("Strides:", tensor.stride())