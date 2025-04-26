############-------PyTorch-------################
import torch
import torch.nn.functional as F

# 使用arange创建连续数据，确保最后一维是偶数以便分割
x_torch = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 10.0 - 3.0
print("PyTorch tensor:")
print(x_torch)

# SwiGLU实现：将tensor在最后一维分成两半
x1, x2 = torch.split(x_torch, x_torch.size(-1) // 2, dim=-1)
out_torch = F.silu(x1) * x2  # SwiGLU: swish(x1) * x2
print("\nPyTorch swiglu result:")
print(out_torch)

############-------DEEPX-------################
from deepx import arange,swish,swiglu

# 使用相同的初始化方式
x = arange(0,48,1,name="x").reshape_(3,4,4)
x.div_(10.0)
x.sub_(3.0)

print("\nDEEPX tensor:")
print(x)

out = swiglu(x,out="out")
print("\nDEEPX swiglu result:")
print(out)
 
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=out.graph.to_dot()
str.render(script_name+".dot", format='svg')