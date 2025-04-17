from deepx.tensor import Tensor

def copytest():
    from deepx.nn.functional import newtensor
    t1= newtensor(1, 2, 3,name='t1')
    t2= newtensor(1, 2, 3,name='t2')
    t1.copy_to(t2)
    print(t2)


if __name__ == "__main__":
    copytest()