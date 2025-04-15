import sys 
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import Tensor
def printall(t):
   print("t=",t)
   print("t.shape=",t.shape)
   print("t.shape[0]=",t.shape[0])
   print("t.stride=",t.stride)
   print("t.stride[0]=",t.stride[0])
   print("t.dim=",t.dim())
   print("t.ndimension=",t.ndimension)
   print("t.numel=",t.numel())
   print("t.dtype=", t.dtype)
def newtensorwithshape(shape):
   t = Tensor(shape=[2,3,4])
   from deepx.nn.functional import newtensor
   newtensor(t)
   printall (t)


def newtensorwithdata():
   t = Tensor([1,2,3])
   from deepx.nn.functional import newtensor
   newtensor(t)
   printall(t)

if __name__ == "__main__":
   args = sys.argv[1:]
   caseid = 0
   if len(args) > 0:
      caseid = int(args[0])
   if caseid == 0:
      newtensorwithshape([1, 2, 3])
   elif caseid == 1:
      newtensorwithdata()
   elif caseid == 2:
      newtensorwithshape([1, 2, 3])