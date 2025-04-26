
############-------PyTorch-------################
print()

import torch
torch_t1 = torch.arange(3*4*5, dtype=torch.float32)
torch_t2 = torch.full((3*4*5,),2, dtype=torch.float32)

torch_t3 = torch.sqrt(torch_t1)
print(torch_t3)
torch_t4 = torch.log(torch_t2)
print(torch_t4)
torch_t5 = torch.exp(torch_t4)
print(torch_t5)
torch_t6 = torch.pow(torch_t5,torch_t3)
print(torch_t6)
torch_t7 = 2**torch_t1
print(torch_t7)
############-------DEEPX-------################

import deepx
print()

t1 = deepx.arange(start=0,end=3*4*5,dtype='float32',name="t1")
t2 = deepx.full((3*4*5,),value=2,dtype='float32',name="t2")
t3 = deepx.sqrt(t1,out='t3')
t3.print()
t4 = deepx.log(t2,out='t4')
t4.print()
t5 = deepx.exp(t4,out='t5')
t5.print()
t6 = deepx.pow(t5,t3,out='t6')
t6.print()
t7 = 2**t1
t7.print()

