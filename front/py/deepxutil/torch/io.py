from torch import Tensor as torch_Tensor

def save_torch(t,path:str):
    r'''
    保存torch.Tensor为deepx.tensor格式
    '''
    assert isinstance(t,torch_Tensor)
    t=t.detach().cpu().numpy()
    from deepxutil.numpy.io import save_numpy
    save_numpy(t,path)
    