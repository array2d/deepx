from deepx.tensor import Tensor
from deepx.nn import DeepxIR
from deepx.scheduler import send

def matmul(a:Tensor,b:Tensor,out: Tensor ,author='cublas'):
    ir=DeepxIR("matmul", [a.node.name,b.node.name], [out.node.name], author)
    send(ir)
    return out