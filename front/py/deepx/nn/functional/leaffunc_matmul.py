from typing import Union

from deepx import Tensor
from deepx.autograd import Function,Context
from .leaffunc_new import newtensor
 
class Matmul(Function):
    @staticmethod
    def forward(ctx:Context,
                a:Tensor,
                b: Tensor, 
                out:Union[Tensor,str]='',
                authormap:dict={'matmul':'cublas'}):
        ctx.save_tensors(a,b)
        ctx.set_authormap(authormap)
 
        outtensor=None
        if isinstance(out,str):
            matmulshape=a.Shape.matmul(b.shape)
            outtensor=newtensor(matmulshape, dtype=a.dtype,name=out)
        else:
            outtensor=out

        from .rtf_matmul import rtf_matmul
        rtf_matmul(a,b,outtensor,ctx.authormap['matmul'])
        return outtensor
    
    @staticmethod
    def backward(ctx:Context,out_grad:Tensor,a_grad:Union[Tensor,str],b_grad:Union[Tensor,str]):
        a,b=ctx.get_tensors()
        if isinstance(a_grad,str):
            a_grad=newtensor(shape=a.shape,dtype=a.dtype,name=a_grad)
        if isinstance(b_grad,str):
            b_grad=newtensor(shape=b.shape,dtype=b.dtype,name=b_grad)
        from .rtf_matmul import rtf_matmul_backward
        rtf_matmul_backward(out_grad,a,b,a_grad,b_grad,ctx.authormap['matmul'])
        return a_grad,b_grad

def matmul(a:Tensor,b:Tensor,out:Union[Tensor,str]='',author:str='cublas')->Tensor:
    return Matmul.apply(a,b,out,author)
