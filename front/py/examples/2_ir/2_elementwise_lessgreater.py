############-------PyTorch-------################

print()
import torch
torch_t1 = torch.full((2,3,4, ), 10, dtype=torch.float32)
torch_t2 = torch.arange(24,dtype=torch.float32).reshape(2,3,4)
torch_t3= torch.less(torch_t2,torch_t1)
print("t1<t2")
print(torch_t3)
torch_t4= torch.greater(torch_t2,torch_t1)
print("t1>t2")
print(torch_t4)
torch_t5= torch.equal(torch_t2,torch_t1)
print("t1==t2")
print(torch_t5)
torch_t6= torch.not_equal(torch_t2,torch_t1)
print("t1!=t2")
print(torch_t6)


############-------DEEPX-------################

from deepx import Tensor,full,arange,less,greater

print()

t1 = full((2,3,4), value=10,dtype="float32")
equalmask=t1==10
equalmask.print()
t2 = arange(0,24,dtype="float32").reshape_((2,3,4))
t3_= t2<t1
t3_.print()
t4_= t2>t1
t4_.print()

t5_= t2==t1
t5_.print()
t6_= t2!=t1
t6_.print()
