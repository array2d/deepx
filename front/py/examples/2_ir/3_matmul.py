############-------PyTorch-------################

import torch
torch_t1 = torch.ones(3, 4, dtype=torch.float32)
torch_t2 = torch.ones(4, 5, dtype=torch.float32)
torch_t3 = torch_t1 @ torch_t2
print()
print(torch_t3)

############-------DEEPX-------################

from deepx import zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32',name="t1")
t2 = ones([4,5],dtype='float32',name="t2")
t3 = t1 @ t2
print(t3)



 