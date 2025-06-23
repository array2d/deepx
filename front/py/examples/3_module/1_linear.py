############-------PyTorch-------################
import torch
import torch.nn as nn

net = nn.Linear(64, 4)
torch_input = torch.ones(1, 64)
torch_output = net(torch_input)
print()
print(torch_output)


############-------DEEPX-------################
from deepx.nn.modules import Linear
from deepx import ones

net = Linear(64, 4)
input=ones((1,64),name='input')
out=net.forward(input)
out.print()

