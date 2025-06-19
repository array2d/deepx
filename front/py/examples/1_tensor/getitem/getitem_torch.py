import torch

t = torch.full((2, 3, 4), 1)
t2 = t[None, :, None]
print(t2)