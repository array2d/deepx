#######====PYTORCH======########

print()
import torch
a=torch.arange(4*2*3).reshape(4,2,3)
b=torch.arange(2*1).reshape(2,1)
bb_torch = torch.broadcast_to(b, (4,2,3))
print(bb_torch)

########====DEEPX====########
from deepx import Tensor,arange,broadcastTo

a=arange(start=0,end=4*2*3,name="a").reshape_((4,2,3))
b=arange(start=0,end=2,name='b').reshape((2,1))
bb=b.broadcastTo( a.shape,out="b.broadcasted")
bb.print()


 

