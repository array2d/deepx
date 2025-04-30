############-------PyTorch-------################

print()
import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.int8)
torch_t2 = torch.arange(24,dtype=torch.int8).reshape(2,3,4)
torch_t3= torch.min(torch_t2,torch_t1)
print(torch_t3)
torch_t4= torch.max(torch_t2,torch_t1)
print(torch_t4)


############-------DEEPX-------################

from deepx import Tensor,full,arange,min,max

print()

t1 = full((2,3,4), value=10,dtype="int8")
t2 = arange(0,24,dtype="int8").reshape_((2,3,4))
t3 = min(t2,t1)
t3.print()
t4 = max(t2,t1)
t4.print()