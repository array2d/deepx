
############-------PyTorch-------################

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

############-------DEEPX-------################

import deepx
print()

t1 = deepx.arange(3*4*5,dtype='float32',name="t1")
t2 = deepx.full([3*4*5],value=2,dtype='float32',name="t2")
t3 = deepx.sqrt(t1,out='t3')
print(t3)
t4 = deepx.log(t2,out='t4')
print(t4)
t5 = deepx.exp(t4,out='t5')
print(t5)
t6 = deepx.pow(t5,t3,out='t6')
print(t6)



