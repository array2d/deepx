############-------PyTorch-------################
import torch

# 使用arange创建连续数据
x_torch = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 10.0 - 3.0
print("PyTorch tensor:")
print(x_torch)

out_torch = torch.softmax(x_torch,-2)
print("\nPyTorch sigmoid result:")
print(out_torch)

import os
dir = os.path.expanduser('~/model/deepxmodel/functional/')
from deepxutil.torch import save_torch
save_torch(x_torch, dir + 'forsoftmax')

############-------DEEPX-------################
from deepx import softmax,load

# 使用相同的初始化方式
x=load(dir+'forsoftmax')

print("\nDEEPX tensor:")
x.print()

out=softmax(x,[-2])
print("\nDEEPX sigmoid result:")
out.print()
