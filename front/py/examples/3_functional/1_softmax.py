############-------PyTorch-------################
import torch

# 使用arange创建连续数据
x_torch = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 10.0 - 3.0
print("PyTorch tensor:")
print(x_torch)

out_torch = torch.sigmoid(x_torch)
print("\nPyTorch sigmoid result:")
print(out_torch)

############-------DEEPX-------################
from deepx import Tensor,ones,zeros,arange
from deepx import sigmoid

# 使用相同的初始化方式
x = arange(3,4,5,name="x")
x.div_(10.0)
x.sub_(3.0)

print("\nDEEPX tensor:")
print(x)

out=sigmoid(x)
print("\nDEEPX sigmoid result:")
print(out)
