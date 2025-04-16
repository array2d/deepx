import sys 
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import Tensor

def newtensor():

   from deepx.nn.functional import newtensor
   t=newtensor(1,2,3,name='t')
   print(t)

if __name__ == "__main__":
   newtensor()
