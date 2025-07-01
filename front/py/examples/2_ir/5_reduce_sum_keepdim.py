############-------PyTorch-------################
print()

import torch
torch_t = torch.arange(0,60).reshape(3,4,5)
print(torch_t)
torch_s = torch.sum(torch_t, dim=[0, 2],keepdim=True)
print(torch_s)
torch_p=torch.prod(torch_t,dim=1)
print(torch_p)

torch_t1 = torch.ones(4, 5, 6,dtype=torch.float)
print(torch_t1)
torch_t2 = torch.sum(torch_t1, dim=[0, 1],keepdim=True)
print(torch_t2)


############-------DEEPX-------################

from deepx import  ones,arange
from deepx.nn.functional import sum,prod

t=arange(0,60,name='t').reshape_((3,4,5))
t.print()
s=sum(t,dim=(0,2),out="s",keepdim=True)

s.print()
p=prod(t,dim=(1,),out="p",keepdim=True)

p.print()

t1=ones((4,5,6),name="t1")

t1.print()
t2=sum(t1,dim=(0,1),out='t2',keepdim=True)

t2.print()
