
def deepx_getitem():
    from deepx  import newtensor
    t=newtensor((2,3,4)).full_(1)
    t2=t[None, :, None]
    t2.print()
def torch_getitem():
    import torch
    t=torch.full((2,3,4),1)
    t2=t[None, :, None]
    print(t2)
if __name__ == "__main__":
    deepx_getitem()
    torch_getitem()