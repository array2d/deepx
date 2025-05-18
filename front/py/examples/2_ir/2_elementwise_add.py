print()
############-------PyTorch-------################

import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch_t1.clone()
torch_t3 = torch_t1 + torch_t2
torch_t3.add_(0.5)

print(torch_t3)
torch_t4 = torch.full((2,3,4), 1.5, dtype=torch.float32)
torch_t5 = 2-torch_t4
print(torch_t5)

############-------DEEPX-------################

from deepx import  full

print()

t1 = full((2,3,4), value=10,dtype="float32")
t2 = t1.clone()
t3 = t1+t2
t3.add_(0.5)
t3.print()

t4 = full((2,3,4), value=1.5,dtype="float32")
t5 = 2-t4
t5.print()