from deepx import newtensor,arange
t = newtensor((64,))
t.arange_()
print()
t2 = t[None, :, None]
t2.print()