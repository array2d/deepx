import torch
from torch import Tensor as torch_Tensor

def save_torch(t,path:str):
    r'''
    保存torch.Tensor为deepx.tensor格式
    '''
    assert isinstance(t,torch_Tensor)
    t=t.detach().cpu().contiguous()
    realdtype=t.dtype
    if t.dtype is torch.bfloat16:
        t=t.view(torch.uint16)
    elif t.dtype is torch.float8_e4m3fn:
        t=t.view(torch.uint8)
    t = t.numpy()
    from deepxutil.numpy.io import save_numpy
    save_numpy(t,path,str(realdtype)[6:])
    