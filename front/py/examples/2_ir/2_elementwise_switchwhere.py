############-------PyTorch-------################

print()
import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch.arange(24,dtype=torch.float32).reshape(2,3,4)
torch_t3= torch.where(torch_t2<torch_t1,torch_t1,torch_t2)
print("t3=where(t2<t1,t1,t2)")
print(torch_t3)


############-------DEEPX-------################

from deepx import Tensor,full,arange,where

print()

t1 = full((2,3,4), value=10,dtype="float32")
t2 = arange(0,24,dtype="float32").reshape_((2,3,4))
t3= where(t2<t1,t1,t2)
t3.print()

