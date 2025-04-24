############-------PyTorch-------################
import numpy as np  
print()
indices_np = np.array([[0, 1, 2], [0, 1, 2]])

print(indices_np)

import torch
torch_t = torch.arange(10*5, dtype=torch.float32).reshape(10,5)
torch_indices = torch.tensor(indices_np)
torch_t2 = torch.index_select(torch_t, 1,torch_indices)
print(torch_t2.shape)
print(torch_t2)


############-------DEEPX-------################

from deepx import Tensor,arange,Shape,load
from deepxutil.numpy import save_numpy

save_numpy(indices_np,'/home/lipeng/model/deepxmodel/tester/testindices')

t = arange(start=0,end=10*5,dtype='float32',name='t').reshape_((10,5))
indices = load('/home/lipeng/model/deepxmodel/tester/testindices')
indices.print()
t2 = t.indexselect(indices,axis=1)
t2.print()

### indexselect 行为和tensorflow.gather保持一致，支持index为多维