print()
############-------PyTorch-------################

import torch
torch_t1 = torch.arange(60, dtype=torch.float32).reshape(3, 4,5)
print(torch_t1)
torch_t2=torch_t1.repeat([1,2,3])
print(torch_t2)


############-------Deepx-------################

from deepx import arange
t1 =  arange(0,60).reshape_((3, 4,5))
t1.print()
t2=t1.repeat((1,2,3))
t2.print()