############-------PyTorch-------################

print()
import torch
torch_t1 = torch.ones(3, 4,5, dtype=torch.float32)
torch_t2 = torch.ones(3, 4,5, dtype=torch.float32)
torch_t3 = torch.ones(3, 4,5, dtype=torch.float32)
 
torch_t = torch.concat([torch_t1, torch_t2, torch_t3], dim=1)
print(torch_t)
 

############-------DEEPX-------################

from deepx import Tensor,zeros, ones, concat


t1 = ones([3,4,5],dtype='float32',name='t1')
t2=ones([3,4,5],dtype='float32',name='t2')
t3=ones([3,4,5],dtype='float32',name='t3')
 
t=concat([t1,t2,t3],dim=1,out='t')
t.print()
