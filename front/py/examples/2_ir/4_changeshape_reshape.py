
############-------PyTorch-------################

import torch
torch_t1 = torch.ones(3, 4, dtype=torch.float32)
print(torch_t1)
torch_t2 = torch_t1.reshape(3, 2, 2)
print(torch_t2)

torch_t3=torch.ones(4, 5, dtype=torch.float32).reshape(-1)
print(torch_t3)

############-------DEEPX-------################

from deepx import Tensor,zeros, ones, full, arange
print()
t1 = ones([3,4],dtype='float32',name='t1')
print(t1)
t2=t1.reshape(3,2,2)
print(t2)

t3=ones([4,5],dtype='float32').reshape_(20)
print(t3)
