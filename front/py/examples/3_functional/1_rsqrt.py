############-------PyTorch-------################

import torch
import torch.nn.functional as F
torch_t = torch.arange(0, 24).reshape(2, 3, 4)
torch_rsqrt_t = torch.rsqrt(torch_t)
print(torch_t)
print(torch_rsqrt_t)

############-------DEEPX-------################

from deepx import  arange
from deepx.nn.functional import rsqrt

t=arange(2,3,4,name='t')
print((t))
rsqrt_t=rsqrt(t)
print(rsqrt_t)
