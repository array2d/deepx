############-------PyTorch-------################
 
print()
import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch_t1.to(dtype=torch.bfloat16)
print(torch_t2)
torch_t3 = torch_t2.to(dtype=torch.float32)
print(torch_t3)

############-------DEEPX-------################

from deepx import  full


t1 = full((2,3,4), value=10,dtype="float32")
t2 = t1.to(dtype="bfloat16")
t2.print()
t3 = t2.to(dtype="float32")
t3.print()