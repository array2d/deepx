import sys
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

def newtensor(dtype):
   from deepx.nn.functional import newtensor
   for i in range(0,20):
      t=newtensor((1,20,4096),dtype=dtype)
      # t.print()


if __name__ == "__main__":
   args=sys.argv[1:]
   if len(args)==0:
      newtensor('float32')
   elif len(args)==1:
      newtensor(args[0])
   else:
      print("Usage: python 1_new.py [dtype]")