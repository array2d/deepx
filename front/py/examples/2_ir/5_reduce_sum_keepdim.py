############-------PyTorch-------################

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

from deepx import Tensor,ones,zeros,arange
from deepx.nn.functional import sum,prod

t=Tensor(shape=(3,4,5))
t.addtograph("t")
t.arange_(0,1)
t.set_format("%.0f")
print(t)
s=sum(t,dim=[0,2],out="s",keepdim=True)
s.set_format("%.0f")
print(s)
p=prod(t,dim=[1],out="p",keepdim=True)
p.set_format("%.0f")
print(p)

t1=ones(4,5,6,name="t1")
t1.set_format("%.0f")
print(t1)
t2=sum(t1,dim=[0,1],out='t2',keepdim=True)
t2.set_format("%.0f")
print(t2)


import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]
str=t2.graph.to_dot()
str.render(script_name+".dot", format='svg')