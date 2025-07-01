############-------PyTorch-------################
print()
import torch
torch_t1 = torch.ones(3, 4, dtype=torch.float32)
print(torch_t1)
torch_t2 = torch_t1.transpose(0, 1)
print(torch_t2)

torch_t3 = torch.ones(2, 3, 4, dtype=torch.float32)
torch_t4 = torch_t3.transpose(1, 2)
print(torch_t4)

############-------DEEPX-------################

from deepx import  ones



t1 = ones((3,4),dtype='float32',name='t1')
t1.print()
t2=t1.transpose(0,1,out='t2')
t2.print()

t3=ones((2,3,4),dtype='float32',name='t3')
t4=t3.transpose(1,2,out='t4')
t4.print()
