import sys 
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import Tensor
def printall(t):
   print("t=",t)
   print("t.name",t.name)
   print("t.shape=",t.shape)
   print("t.shape[0]=",t.shape[0])
   print("t.stride=",t.stride)
   print("t.stride[0]=",t.stride[0])
   print("t.dim=",t.dim())
   print("t.ndimension=",t.ndimension)
   print("t.numel=",t.numel())
   print("t.dtype=", t.dtype)

def newtensor():

   from deepx.nn.functional import newtensor
   t=newtensor(1,2,3)
   printall(t)

if __name__ == "__main__":
   newtensor()
