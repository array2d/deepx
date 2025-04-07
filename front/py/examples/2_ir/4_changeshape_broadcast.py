#######====PYTORCH======########


import torch
a=torch.arange(4*2*3).reshape(4,2,3)
b=torch.arange(2*1).reshape(2,1)
bb_torch = torch.broadcast_to(b, (4,2,3))
print(bb_torch)
c_torch=a+bb_torch
print(c_torch)

########====DEEPX====########
from deepx import Tensor,arange,broadcast_to

a=arange(end=4*2*3 ,name="a").reshape_(4,2,3)
b=arange(end=2*1 ,name='b').reshape_(2,1)
bb=b.broadcast_to( a.shape,out="b.broadcasted")
print(bb)
c=a+bb
print(c)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=b.graph.to_dot()
str.render(script_name+".dot", format='svg')
 

