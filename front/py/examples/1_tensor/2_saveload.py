from deepx.nn.functional import arange,save,load

def saveloadfloat32():
    t1=arange(start=0,end=60 ,dtype='float32').reshape_((3,4,5))
    dir='/home/lipeng/model/deepxmodel/tester/'
    t1.save(dir+'t1')

    t2=load(dir+'t1')
    t2.print()

def saveloadint8():
    t=arange(start=0,end=60 ,dtype='int8').reshape_((3,4,5))
    dir='/home/lipeng/model/deepxmodel/tester/'
    t.save(dir+'tint8')

    t2=load(dir+"tint8")
    t2.print()


if __name__ == "__main__":
    #saveloadfloat32()
    saveloadint8()