#######====PYTORCH======########


import torch
a=torch.arange(4*2*3).reshape(4,2,3)
b=torch.arange(2*1).reshape(2,1)
bb_torch = torch.broadcast_to(b, (4,2,3))
print(bb_torch)

########====DEEPX====########
from deepx import Tensor,arange,broadcastTo

a=arange(4,2,3,name="a")
b=arange(2,1,name='b')
bb=b.broadcastTo( a.shape,out="b.broadcasted")
print(bb)


 

