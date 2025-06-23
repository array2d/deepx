#######====PYTORCH======########

print()
import torch
torch_a=torch.arange(4*2*3).reshape(4,2,3)
torch_b=torch.arange(2*1).reshape(2,1)
bb_torch = torch.broadcast_to(torch_b, (4,2,3))
print(bb_torch)
torch_a[None:,]



########====DEEPX====########
from deepx import  arange

a=arange(start=0,end=4*2*3,name="a").reshape_((4,2,3))
b=arange(start=0,end=2,name='b').reshape((2,1))
bb=b.unsqueeze(0).broadcastTo(a.shape,out="b.broadcasted")
bb.print()

c=a[None:,]

