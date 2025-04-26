############-------PyTorch-------################

import torch
torch_t = torch.arange(0, 24,dtype=torch.float).reshape(2, 3, 4)
torch_rsqrt_t = torch.rsqrt(torch_t)
print(torch_t)
print(torch_rsqrt_t)

import os
dir = os.path.expanduser('~/model/deepxmodel/functional/')
from deepxutil.torch import save_torch
save_torch(torch_t, dir + 'aranged')

############-------DEEPX-------################

from deepx import  rsqrt,load

t=load(dir+'aranged')
t.print()
rsqrt_t=rsqrt(t)
rsqrt_t.print()
