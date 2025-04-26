############-------PyTorch-------################
print()

import torch
import torch.nn.functional as F
torch_t = torch.empty(10, 10).uniform_(-1, 1)
torch_relu_t = F.relu(torch_t)
print(torch_t)
print(torch_relu_t)

import os
dir=os.path.expanduser('~/model/deepxmodel/functional/')
from deepxutil.torch import save_torch
save_torch(torch_t,dir+'uniformed')
 
############-------DEEPX-------################

from deepx  import relu,load


t=load(dir+'uniformed')
t.print()
relu_t=relu(t)
relu_t.print()

