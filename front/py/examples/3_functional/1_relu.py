############-------PyTorch-------################

import torch
import torch.nn.functional as F
torch_t = torch.empty(10, 10).uniform_(-1, 1)
torch_relu_t = F.relu(torch_t)
print(torch_t)
print(torch_relu_t)

############-------DEEPX-------################

from deepx import Tensor,ones
from deepx.nn.functional import relu,uniform


t=uniform(10,10,low=-1,high=1,name='t')

print(t)
relu_t=relu(t)
print(relu_t)

# 当tensor.name为str时，说明其是中间变量，执行inplace操作
t2=uniform(10,10,low=-1,high=1)
print(t2)
relu_t2=relu(t2)
print(relu_t2)

