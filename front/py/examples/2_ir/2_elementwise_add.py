############-------PyTorch-------################

import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch_t1.clone()
torch_t3 = torch_t1 + torch_t2
torch_t3.add_(0.5)
print()
print(torch_t3)

############-------DEEPX-------################

from deepx import  full

print()

t1 = full(2,3,4, value=10,dtype="float32")
t2 = t1.clone()
t3 = t1+t2
t3.add_(0.5)
t3.print()