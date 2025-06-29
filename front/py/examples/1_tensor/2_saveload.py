from deepx.nn.functional import arange,save,load,full

dir = '/home/lipeng/model/deepx/tester/'

def saveload(dtype:str='float32'):
    print()


    t1=full((3,4,5),2,dtype)
    # t1=arange(start=0,end=60 ,dtype=dtype)
    # t1=t.reshape_((3,4,5))
    t1.float().print()
    t1.print()
    name='t_'+dtype
    t1.save(dir+name)
    t2=load(dir+name)
    t2.print()

if __name__ == "__main__":
    saveload("bfloat16")