from deepx.tensor import Tensor
from deepx.nn import DeepxIR,Param
from deepx.scheduler import send
from .rtf import A_B_op_C

def rtf_matmul(a:Tensor,b:Tensor,out: Tensor ,author='cublas'):
    args=[Param.tensor(a),Param.tensor(b)]
    returns=[Param.tensor(out)]
    ir=DeepxIR("matmul", args, returns, author)
    send(ir)
    return out