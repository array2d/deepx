############-------PyTorch-------################

import torch
torch_t = torch.arange(100*3, dtype=torch.float32).reshape_(100,3) 
torch_indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
torch_t = torch.gather(torch_t, torch_indices, dim=1)
print(torch_t)


############-------DEEPX-------################

from deepx import Tensor,zeros, ones, concat

t = Tensor.arange(100,3,dtype='float32',name='t')

indices = Tensor([[0, 1, 2], [0, 1, 2]],dtype='int32',name='indices')
t = t.gather(indices,dim=1)
print(t)