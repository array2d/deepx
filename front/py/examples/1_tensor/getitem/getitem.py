from deepx import newtensor,arange
t = newtensor((2, 3, 13))
t.arange_()
print()
t2 = t[None, :, None]
t2.print()
t3=t[:,None,:]
t3.print()
x=t
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
x1.print()
x2.print()
