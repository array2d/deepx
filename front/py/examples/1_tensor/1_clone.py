
from deepx  import Tensor,newtensor,rnewtensor

def clonetest():
    t1=Tensor(shape=(1,2,3),dtype='float32',name='t1')
    rnewtensor(t1)
    t2=t1.clone(name='t2')
    t2.print()

if __name__ == "__main__":
    clonetest()