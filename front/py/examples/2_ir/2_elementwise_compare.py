############-------PyTorch-------################

print()
import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.int8)
torch_t2 = ~torch_t1
print(torch_t2)
torch_t3 = torch.full((2,3,4, ), 2, dtype=torch.int64)
torch_t4 = ~torch_t3
print(torch_t4)



############-------DEEPX-------################

from deepx import Tensor,full

print()

t1 = full((2,3,4), value=10,dtype="int8")
t2 = ~t1
t2.print()

t3 = full((2,3,4), value=2,dtype="int64")
t4 = ~t3
t4.print()