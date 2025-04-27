############-------PyTorch-------################

print()
import torch
torch_t1 = torch.arange(24, dtype=torch.int32).reshape(2,3,4)
torch_t2 = torch_t1.dropout(p=0.5)
print(torch_t2)
 



############-------DEEPX-------################

from deepx import Tensor,arange

print()

t1 = arange(start=0,end=24 ,dtype="int32").reshape_(2,3,4)
t2 = t1.dropout(p=0.5)
t2.print()
 