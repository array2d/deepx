from deepx import newtensor,Tensor
t = newtensor((2, 3, 13))
t.full_(1)
print()
t2 = t[None, :, None]
t2.print()
t3=t[:,None,:]
t3.print()
t4=t[..., : t.shape[-1] // 2]
t4.print()
