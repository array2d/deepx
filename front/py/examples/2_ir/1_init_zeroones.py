
############-------PyTorch-------################

import torch
torch_t1 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch_t2 = torch.ones(3, 4, 5, dtype=torch.float32)
torch_t3 = torch_t1 + torch_t2
torch_t4 = torch.full((3, 4, 5), 0.5)
print(torch_t3)
torch_t5 = torch_t4 + torch_t3
print(torch_t5)

torch_t6 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch.nn.init.kaiming_uniform_(torch_t6)
print(torch_t6)



############-------DEEPX-------################

from deepx import zeros,ones,full,kaiming_uniform_
print()

t1 = zeros([3,4,5],dtype='float32',name="t1")
t2 = ones([3,4,5],dtype='float32',name="t2")
t3 = t1.add(t2,"t3")
t4=full([3,4,5],value=0.5,name="t4")
print(t3)
t5=t4.add(t3,"t5")
print(t5)


t6=zeros([3,4,5],dtype='float32')
kaiming_uniform_(t6)
print(t6)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')
