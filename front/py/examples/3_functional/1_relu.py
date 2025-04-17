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

print((t))
relu_t=relu(t,out='relu_t')
print(relu_t)


