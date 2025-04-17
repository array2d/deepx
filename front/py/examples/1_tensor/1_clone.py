from deepx.tensor import Tensor

def clonetest():
    t1=Tensor(shape=(1,2,3),dtype='float32',name='t1')
    t2=t1.clone(name='t2')
    print(t2)

if __name__ == "__main__":
    clonetest()