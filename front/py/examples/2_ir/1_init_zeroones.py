
############-------PyTorch-------################

import torch
torch_t1 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch_t2 = torch.ones(3, 4, 5, dtype=torch.float32)
torch_t4 = torch.full((3, 4, 5), 0.5)
print(torch_t4)

torch_t6 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch.nn.init.kaiming_uniform_(torch_t6)
print(torch_t6)



############-------DEEPX-------################

from deepx import zeros,ones,full,kaiming_uniform
print()

t1 = zeros([3,4,5],dtype='float32')
t2 = ones([3,4,5],dtype='float32')
t4=full([3,4,5],value=0.5)
print(t4)

t6=kaiming_uniform(3,4,5,dtype='float32')
print(t6)
