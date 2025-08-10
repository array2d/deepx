from deepx.tensor import Tensor
from deepx.nn import DeepxIR,Param
from deepx.scheduler import send


def rtf_matmul(a:Tensor,b:Tensor,out: Tensor ,author='cublas',bench:int=None):
    args=[Param.tensor(a),Param.tensor(b)]
    returns=[Param.tensor(out)]
    ir=DeepxIR("matmul", args, returns, author)
    if bench is not None:
        ir._metadata.openbench(bench)
    send(ir)
    return out

@deepx_op("matmul")
def rtf_matmul_2(a:Tensor,b:Tensor,out: Tensor ,author='cublas',bench:int=None):
    return out