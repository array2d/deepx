############-------PyTorch-------################
print()

import torch
import torch.nn.functional as F
torch_t = torch.empty(10, 10).uniform_(-1, 1)
torch_relu_t = F.dropout(torch_t)
print(torch_t)
print(torch_relu_t)
