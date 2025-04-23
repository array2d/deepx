from deepx.tensor import Tensor
from .rtf import A_b1_b2_op_C

def rtf_sum(a:Tensor,dim:tuple[int],keepdim:bool,out: Tensor, author:str='miaobyte')->Tensor:
    A_b1_b2_op_C("sum",a,dim,keepdim,out,author)
 
    
def rtf_prod(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    A_b1_b2_op_C("prod",a,dim,keepdim,out,author)
 

def rtf_reducemax(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    A_b1_b2_op_C("reducemax",a,dim,keepdim,out,author)
 

def rtf_reducemin(a:Tensor,dim:tuple[int],keepdim:bool,out:Tensor, author:str='miaobyte')->Tensor:
    A_b1_b2_op_C("reducemin",a,dim,keepdim,out,author)
 