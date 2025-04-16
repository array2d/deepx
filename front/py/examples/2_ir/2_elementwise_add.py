############-------PyTorch-------################

import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch.full((2,3,4, ), 5, dtype=torch.float32)
torch_t3 = torch_t1 + torch_t2
torch_t3.add_(0.5)
print()
print(torch_t3)

############-------DEEPX-------################

from deepx import Tensor,full

print()

t1 = full(2,3,4, value=10,dtype="float32")
print(t1)
t2 = full(2,3,4, value=5,dtype="float32")
print(t2)
t3 = t1+t2
print(t3)
t3.add_(0.5)
print(t3)
