import torch

t = torch.full((2, 3, 13), 1)
t2 = t[None, :, None]
print(t2.shape)
print(t2)
x=t
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
print(x1)
print(x2)
