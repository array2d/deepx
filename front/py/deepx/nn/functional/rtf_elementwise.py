from deepx.tensor import Tensor,Number
from deepx.nn.deepxir import DeepxIR,Param
from deepx.scheduler import send
from .rtf import A_B_op_C,A_B_c_op_D,A_scalar_op_C,A_scalar_c_op_D,A_op_C

# 四则运算
def rtf_add(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("add",a,b,out,author)
    return out

def rtf_addscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("addscalar",a,b,out,author)
    return out

def rtf_sub(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("sub",a,b,out,author)
    return out

def rtf_subscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("subscalar",a,b,out,author)
    return out

def rtf_rsubscalar(a:float, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    args = [ Param.varnum(a),Param.tensor(b)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("rsubscalar", args, returns, author)
    send(ir)
    return out

def rtf_mul(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("mul",a,b,out,author)
    return out

def rtf_mulscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("mulscalar",a,b,out,author)
    return out

def rtf_div(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("div",a,b,out,author)
    return out

def rtf_divscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("divscalar",a,b,out,author)
    return out

def rtf_rdivscalar(a:float, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    args = [ Param.varnum(a),Param.tensor(b)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("rdivscalar", args, returns, author)
    send(ir)
    return out

# 幂、指数 运算
def rtf_sqrt(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("sqrt",a,out,author)
    return out

def rtf_pow(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("pow",a,b,out,author)
    return out

def rtf_powscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("powscalar",a,b,out,author)
    return out

def rtf_rpowscalar(a:Number,b:Tensor,out:Tensor,author='miaobyte')->Tensor:
    args = [ Param.varnum(a),Param.tensor(b)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("rpowscalar", args, returns, author)
    send(ir)
    return out

def rtf_exp(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("exp",a,out,author)
    return out

def rtf_log(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("log",a,out,author)
    return out

# 三角函数
def rtf_sin(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("sin",a,out,author)
    return out

def rtf_cos(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("cos",a,out,author)
    return out

def rtf_tan(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("tan",a,out,author)
    return out

# 取大小值
def rtf_max(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("max",a,b,out,author)
    return out

def rtf_maxscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("maxscalar",a,b,out,author)
    return out

def rtf_min(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("min",a,b,out,author)
    return out

def rtf_minscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("minscalar",a,b,out,author)
    return out

# 位运算
def rtf_invert(a:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_op_C("invert",a,out,author)
    return out

#比较
# A<B -> C 等价于 B>=A -> C
def rtf_less(a:Tensor, b:Tensor,out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("less",a,b,out,author)
    return out
# A>B -> C
def rtf_greater(a:Tensor, b:Tensor, out:Tensor, author='miaobyte')->Tensor:
    A_B_op_C("greater",a,b,out,author)
    return out
# A<b -> C 等价于 b>=A -> C
def rtf_lessscalar(a:Tensor, b:float,out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("lessscalar",a,b,out,author)
    return out
# A>b -> C
def rtf_greaterscalar(a:Tensor, b:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_op_C("greaterscalar",a,b,out,author)
    return out

# A==B -> C
def rtf_equal(a:Tensor, b:Tensor,epsilon:float, out:Tensor, author='miaobyte')->Tensor:
    A_B_c_op_D("equal",a,b,epsilon,out,author)
    return out
# A==b -> C
def rtf_equalscalar(a:Tensor, b:float,epsilon:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_c_op_D("equalscalar",a,b,epsilon,out,author)
    return out
# A!=B -> C
def rtf_notequal(a:Tensor, b:Tensor,epsilon:float, out:Tensor, author='miaobyte')->Tensor:
    A_B_c_op_D("notequal",a,b,epsilon,out,author)
    return out
# A!=b -> C
def rtf_notequalscalar(a:Tensor, b:float,epsilon:float, out:Tensor, author='miaobyte')->Tensor:
    A_scalar_c_op_D("notequalscalar",a,b,epsilon,out,author)
    return out

# 根据cases[index]的值tensoridx，从X[tensoridx]这个Tensor[index]，赋值给out[index]
def rtf_switch(X:tuple[Tensor,...], cases:Tensor, out:Tensor, author='miaobyte')->Tensor:
    args = [Param.listtensor(X),Param.tensor(cases)]
    returns = [Param.tensor(out)]
    ir = DeepxIR("switch", args, returns, author)
    send(ir)
    return out


# 类型转换
def rtf_todtype(t:Tensor,dest:Tensor):
    assert isinstance(t,Tensor)
    assert isinstance(dest,Tensor)
    assert t.shape==dest.shape

    args=[Param.tensor(t)]
    returns=[Param.tensor(dest)]
    ir=DeepxIR("todtype", args, returns,'')
    send(ir)

