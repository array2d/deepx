############-------PyTorch-------################
print()

import torch
import torch.nn.functional as F
torch_t = torch.empty(10, 10).uniform_(-1, 1)
torch_dropout_t = F.dropout(torch_t)
print(torch_t)
print(torch_dropout_t)


############-------Deepx-------################

from deepx import uniform
t = uniform((10, 10), -1, 1)
dropout_t = t.clone()
dropout_t.dropout_(0.5)
dropout_t.print()
