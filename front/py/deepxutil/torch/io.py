def save_torch(t,path:str):
    r'''
    保存torch.Tensor为deepx.tensor格式
    '''
    from torch import Tensor as torch_Tensor
    assert isinstance(t,torch_Tensor)
    t=t.detach().cpu().numpy()
    from deepxutil.numpy.io import save_numpy
    save_numpy(t,path)
    