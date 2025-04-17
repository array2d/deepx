############-------PyTorch-------################
import torch

# 使用arange创建连续数据
x_torch = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 10.0 - 3.0
print("PyTorch tensor:")
print(x_torch)

out_torch = torch.softmax(x_torch,-1)
print("\nPyTorch sigmoid result:")
print(out_torch)

############-------DEEPX-------################
from deepx import Tensor,ones,zeros,arange
from deepx import softmax

# 使用相同的初始化方式
x = arange(3,4,5,name="x")
x.div_(10.0)
x.sub_(3.0)

print("\nDEEPX tensor:")
x.print()

out=softmax(x,-1)
print("\nDEEPX sigmoid result:")
out.print()
