############-------PyTorch-------################
import os
print()
dir=os.path.expanduser('~/model/deepxmodel/functional/')
import torch
torch_t = torch.arange(10*5, dtype=torch.float32).reshape(10,5)
index=[0, 1, 2,0, 1, 2]
torch_index = torch.tensor(index,dtype=torch.int32)

from deepxutil.torch import save_torch
save_torch(torch_index,dir+'gatherindex')

torch_t2 = torch.index_select(torch_t, 1,torch_index)
print(torch_t2.shape)
print(torch_t2)


############-------DEEPX-------################

from deepx import  arange ,load

t = arange(start=0,end=10*5,dtype='float32',name='t').reshape_((10,5))
indices = load(dir+'gatherindex')
indices.print()
t2 = t.indexselect(indices,axis=1)
t2.print()

### indexselect 行为和tensorflow.gather保持一致，支持index为多维