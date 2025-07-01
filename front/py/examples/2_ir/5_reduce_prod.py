############-------PyTorch-------################

import torch
torch_t = torch.arange(0,60).reshape(3,4,5)
print(torch_t)

torch_p=torch.prod(torch_t,dim=1)
print(torch_p)



############-------DEEPX-------################

from deepx import  arange
from deepx.nn.functional import  prod

t=arange(0,60,name='t').reshape_((3,4,5))
t.print()

p=prod(t,dim=(1,),out="p")
p.print()
