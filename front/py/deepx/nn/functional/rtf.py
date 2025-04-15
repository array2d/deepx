from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send
from typing import Union
def A_B_op_C(op:str,a:Tensor,b:Tensor,out:Tensor,author='miaobyte'):
    args=[Param.tensor(a),Param.tensor(b)]
    returns=[Param.tensor(out)]
    ir=DeepxIR(op, args, returns,author)
    send(ir)

def A_scalar_op_C(op:str,a:Tensor,b:Union[float,int],out:Tensor,author='miaobyte'):
    args=[Param.tensor(a),Param.varnum(b)]
    returns=[Param.tensor(out)]
    ir=DeepxIR(op, args, returns,author)
    send(ir)

def A_op_C(op:str,a:Tensor,out:Tensor,author='miaobyte'):
    args=[Param.tensor(a)]
    returns=[Param.tensor(out)]
    ir=DeepxIR(op, args, returns,author)
    send(ir)

def A_b1_b2_op_C(op:str,a:Tensor,b1:tuple[int],b2:bool,out:Tensor,author='miaobyte'):
    args=[Param.tensor(a),Param.vector(b1,'int32'),Param.varbool(b2)]
    returns=[Param.tensor(out)]
    ir=DeepxIR(op, args, returns,author)
    send(ir)

