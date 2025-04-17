############-------PyTorch-------################

import torch

torch_t3 = torch.arange(0, 120,dtype=torch.float).reshape(4, 5, 6)
print(torch_t3)
torch_t3_mean = torch.mean(torch_t3, dim=[0, 1])
print(torch_t3_mean)

############-------DEEPX-------################

from deepx import  arange
from deepx.nn.functional import  mean


t3=arange(4,5,6,name="t3")
print(t3)

t3_mean=mean(t3,dim=(0,1))
print(t3_mean)
