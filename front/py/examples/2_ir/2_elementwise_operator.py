
############-------PyTorch-------################

import torch
torch_t1 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch_t2 = torch.ones(3, 4, 5, dtype=torch.float32)
torch_t3 = torch_t1 + torch_t2
torch_t4 = torch.full((3, 4, 5), 0.5)
torch_t5 = torch_t4 + torch_t3
print(torch_t5)
torch_t6 = torch_t1 / torch_t2
print(torch_t6)
torch_t7=0.05/torch_t2*2.5
print(torch_t7)

torch_t8=torch_t7.mul(torch_t2)
print(torch_t8)
############-------DEEPX-------################

import deepx
print()

t1 = deepx.zeros([3,4,5],dtype='float32',name="t1")
t2 = deepx.ones([3,4,5],dtype='float32',name="t2")
t3 = t1.add(t2,out='t3')
t4=deepx.full([3,4,5],value=0.5,name='t4')
t5=t4.add(t3,out='t5')
print(t5)
t6=t1.div(t2,out='t6')
print(t6)
t7=t2.rdiv(0.05,out='t7')
t7.mul_(2.5)
print(t7)
t8=t7.mul(t2,out='t8')
print(t8)
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')


