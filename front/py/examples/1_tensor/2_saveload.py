from deepx.nn.functional import arange,save,load

dir = '/home/lipeng/model/deepx/tester/'

def saveloadfloat32():
    t1=arange(start=0,end=60 ,dtype='float32').reshape_((3,4,5))

    t1.print()
    t1.save(dir+'t1')

    t2=load(dir+'t1')
    t2.print()

def saveloadint8():
    t=arange(start=0,end=60 ,dtype='int8').reshape_((3,4,5))

    t.save(dir+'tint8')

    t2=load(dir+"tint8")
    t2.print()


def saveloadbfloat16():
    t=arange(start=0,end=60 ,dtype='bfloat16').reshape_((3,4,5))
    t.print()
    t.save(dir+'bf16')

    t2=load(dir+"bf16")
    t2.print()


if __name__ == "__main__":
    saveloadfloat32()
    # saveloadint8()
    # saveloadbfloat16()