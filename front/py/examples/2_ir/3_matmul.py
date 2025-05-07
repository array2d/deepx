benchcnt=2000
shape=[4096,4096]
print()
from deepxutil.numpy  import save_numpy
import numpy as np
np_T1 = np.random.randn(shape[0], shape[1]).astype(np.float32)
np_T2 = np.random.randn(shape[0], shape[1]).astype(np.float32)

npy_path = '/home/lipeng/model/deepxmodel/matmul/'
save_numpy(np_T1,npy_path+'t1')
save_numpy(np_T2,npy_path+'t2')

############-------PyTorch-------################

import torch
import time
torch_t1 = torch.from_numpy(np_T1).to(torch.float32).to('cuda')
torch_t2 = torch.from_numpy(np_T2).to(torch.float32).to('cuda')
# warmup
_=torch_t1 @ torch_t2

torch_start = time.time()
for i in range(benchcnt):
    torch_t3 = torch_t1 @ torch_t2

torch_end = time.time()
print(f"PyTorch time: {torch_end - torch_start} seconds")
############-------DEEPX-------################

from deepx import   matmul, zeros,load
print()

t1 = load(npy_path+'t1')
t2 = load(npy_path+'t2')
t3= zeros(tuple(shape),dtype='float32',name="t3")
from deepx.nn.functional import defaultauthor
defaultauthor['matmul']='miaobyte'
# warmup
matmul(t1,t2,out=t3)

deepx_start = time.time()
matmul(t1,t2,out=t3,bench=benchcnt)
deepx_end = time.time()
print(f"DeepX time: {deepx_end - deepx_start} seconds")



 