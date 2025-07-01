
print()
#######-----------------torch-----------------#######
import torch
torch_x = torch.arange(6).reshape(1,2,3)       # shape=(2,3)
torch_y = torch_x.broadcast_to((3,2,3))    # 需要原维度为1
print(torch_y)

torch_x2=torch_x.repeat_interleave(dim=0, repeats=3)
print(torch_x2)


#######-----------------deepx-----------------#######
from deepx import  arange
deepx_x = arange(0,6).reshape_((1,2,3))      # shape=(2,3)
deepx_y = deepx_x.broadcast_to((3,2,3))    # 需要原维度为1
deepx_y.print()






