from typing import Tuple, List, Optional,Union
import time
from datetime import datetime  # 添加datetime模块
from deepx.tensor import Tensor
class Param:
    def __init__(self,textvalue:str, category:str=None,precision:str=None):
        self._textvalue=textvalue
        self._category=category
        self._precision=precision

    def __str__(self):
        if self._category is not None:
            if self._precision is not None:
                return f"{self._category}<{self._precision}> {self._textvalue}"
            else:
                return f"{self._category} {self._textvalue}"
        else:
            return self._textvalue
    
    @classmethod
    def tensorName(cls,name:str,dtype:str):
        return Param(name,category="tensor",precision=dtype)

    @classmethod
    def tensor(cls,t:Tensor):
        tid=id(t)
        name=f"tensor_{tid}"
        return Param(name,category="tensor",precision=t.dtype)


    @classmethod
    def varnum(cls,value:Union[float,int]):
        precision=None
        if isinstance(value,float):
            precision="float"
        elif isinstance(value,int):
            precision="int"
        return Param(str(value),category="var",precision=precision)

    @classmethod
    def varbool(cls,value:bool):
        return Param(str(value),category="var",precision="bool")

    @classmethod
    def varstr(cls,value:str):
        return Param(value,category="var",precision="string")

    @classmethod
    def vector(cls,value:tuple,dtype:str):
        textvalue='['+' '.join(str(v) for v in value)+']'
        return Param(textvalue,category="vector",precision=dtype)
    
    @classmethod
    def tensorlist(cls,value:tuple[str],dtype:str):
        textvalue='['+' '.join(v for v in value)+']'
        return Param(textvalue,category="tensorlist",precision=dtype)
    
class DeepxIR:
    def __init__(self, 
                name:str,
                args: List[Param], 
                returns: List[Param],
                author:str=''):
        """
        初始化操作节点
        Args:
            args: 输入参数名称列表,如["input", "weight"]
            returns: 输出参数名称列表,如["output"]
            author: tensorfunc的作者名称,如"miaobyte"
        """
 
        self._name = name
        self._args = [arg if isinstance(arg, Param) else Param(arg) for arg in args]
        self._returns = [ret if isinstance(ret, Param) else Param(ret) for ret in returns]
        self._author = author
        self._id=None
        self._created_at=time.time()
        self._sent_at=None

    def __str__(self):
        # 函数名部分
        parts = [self._name]
        
        # 处理输入参数部分 - 使用括号和逗号分隔
        args_parts = []
        for arg in self._args:
            args_parts.append(str(arg))
        
        # 添加输入参数括号和逗号分隔
        parts.append("(" + ", ".join(args_parts) + ")")
        
        # 添加箭头
        parts.append("->")
        
        # 处理输出参数部分 - 使用括号和逗号分隔
        returns_parts = []
        for ret in self._returns:
            returns_parts.append(str(ret))
        
        # 添加输出参数括号和逗号分隔
        parts.append("(" + ", ".join(returns_parts) + ")")

        # 添加元数据
        parts.append("//")
        if self._id is not None:
            parts.append(f"id={self._id}")
        if self._author:
            parts.append(f"author={self._author}")
        parts.append(f"created_at={self._created_at}")
        if self._sent_at is not None:
            parts.append(f"sent_at={self._sent_at}")
        
        return ' '.join(parts)

class DeepxIRResp:
    #'1 ok examplemsg // recv_at=1741494459006 start_at=1741494459006 finish_at=1741494459006'
    def __init__(self,s:str):
        self._id=None
        self._result=""
        self._message=''
        #extra info
        self._recv_at=None  
        self._start_at=None
        self._finish_at=None
        
        # 解析响应字符串
        if s and isinstance(s, str):
            # 首先按 "//" 分割为前后两部分
            parts = s.split("//", 1)
            
            if len(parts) >= 1:
                # 处理前半部分 ID、结果和消息
                front_parts = parts[0].strip().split(" ", 2)
                
                if len(front_parts) >= 1:
                    self._id = front_parts[0]
                
                if len(front_parts) >= 2:
                    self._result = front_parts[1]
                
                if len(front_parts) >= 3:
                    self._message = front_parts[2]
            
            # 处理后半部分的时间戳信息
            if len(parts) >= 2:
                extra_info = parts[1].strip()
                extra_parts = extra_info.split()
                
                for part in extra_parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        if key == "recv_at":
                            # 将毫秒时间戳转换为datetime对象
                            self._recv_at = datetime.fromtimestamp( float(value) / 1000.0)
                        elif key == "start_at":
                            self._start_at =datetime.fromtimestamp( float(value) / 1000.0)
                        elif key == "finish_at":
                            self._finish_at = datetime.fromtimestamp( float(value) / 1000.0)
 
    def __str__(self) -> str:
        parts=[]
        parts.append(self._id)
        parts.append(self._result)
        parts.append(self._message)
        parts.append("//")
        if self._recv_at is not None:
            parts.append(f"recv_at={self._recv_at.strftime('%H:%M:%S.%f')[:-3]}")
        if self._start_at is not None:
            parts.append(f"start_at={self._start_at.strftime('%H:%M:%S.%f')[:-3]}")
        if self._finish_at is not None:
            parts.append(f"finish_at={self._finish_at.strftime('%H:%M:%S.%f')[:-3]}")
        return ' '.join(parts)