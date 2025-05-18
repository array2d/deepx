import sys 
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import Tensor
def printall(t):

   print("t.name",t.name)
   print("t.shape=",t.shape)
   print("t.shape[0]=",t.shape[0])
   print("t.stride=",t.stride)
   print("t.stride[0]=",t.stride[0])
   print("t.dim=",t.dim())
   print("t.ndimension=",t.ndimension)
   print("t.numel=",t.numel())
   print("t.dtype=", t.dtype)
   t.print()

def newtensor(dtype):

   from deepx.nn.functional import newtensor
   t=newtensor((1,2,3),dtype=dtype)
   printall(t)


if __name__ == "__main__":
   args=sys.argv[1:]
   if len(args)==0:
      newtensor('float32')
   elif len(args)==1:
      newtensor(args[0])
   else:
      print("Usage: python 1_new.py [dtype]")
