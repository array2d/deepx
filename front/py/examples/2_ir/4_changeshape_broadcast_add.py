########====DEEPX====########
from deepx import ones

a=ones( 4,2,3 ,name="a")    
b=ones(  2,1 ,name='b')
c=a+b
print(c)

########====pytorch====########
import torch
torch_a=torch.ones(4,2,3)
torch_b=torch.ones(2,1)
torch_c=torch_a+torch_b
print(torch_c)

