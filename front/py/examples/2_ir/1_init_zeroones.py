
############-------PyTorch-------################

import torch
torch_t1 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch_t2 = torch.ones(3, 4, 5, dtype=torch.float32)
torch_t4 = torch.full((3, 4, 5), 0.5)
print(torch_t4)
torch_t5=torch.nn.init.uniform_(torch.zeros(3,4,5),0,1)
print(torch_t5)


torch_t6 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch.nn.init.kaiming_uniform_(torch_t6)
print(torch_t6)

torch_t7 = torch.zeros(3, 4, 5, dtype=torch.float32)
torch_t7.normal_(mean=0,std=0.02)
print(torch_t7)

############-------DEEPX-------################

import deepx
print()

t1 = deepx.zeros((3,4,5),dtype='float32')
t2 = deepx.ones((3,4,5),dtype='float32')
t4=deepx.full((3,4,5),value=0.5)
t4.print()
t5=deepx.uniform((3,4,5),low=0,high=1)
t5.print()
t6=deepx.kaiming_uniform((3,4,5),dtype='float32')
t6.print()

t7=deepx.zeros((3,4,5),dtype='float32')
t7.normal_(mean=0,stddev=0.02)
t7.print("%.6f")
