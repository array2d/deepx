############-------PyTorch-------################
import numpy as np  
print()
indices_np = np.array([[0, 1, 2], [0, 1, 2]])

print(indices_np)

import torch
torch_t = torch.arange(10*5, dtype=torch.float32).reshape(10,5)
torch_indices = torch.tensor(indices_np)
torch_t = torch.gather(torch_t, 1,torch_indices)
print(torch_t.shape)
print(torch_t)


############-------DEEPX-------################

from deepx import Tensor,arange,Shape
from deepx.nn.functional import load,save_npy

 
save_npy(indices_np,'/home/lipeng/model/deepxmodel/tester/testindices')

t = arange(start=0,end=10*5,dtype='float32',name='t').reshape(10,5)
indices = load('/home/lipeng/model/deepxmodel/tester/testindices')
indices.print()
t = t.gather(indices,dim=1)
t.print()