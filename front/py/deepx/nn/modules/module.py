import re
from typing import (Dict, Iterator, Optional, Tuple, Union, 
                    Any, List, overload)
from collections import OrderedDict
from deepx import Tensor

class Module:  
    def __init__(self, name: Optional[str] = None):
        self._name = name or self._generate_default_name()
        self._parent: Optional[Module] = None
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._parameters: OrderedDict[str, Tensor] = OrderedDict()
        self._buffers: OrderedDict[str, Tensor] = OrderedDict()
        self.training = True
        
    def _generate_default_name(self) -> str:
        class_name = self.__class__.__name__
        base_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        if not hasattr(self.__class__, '_instance_counter'):
            self.__class__._instance_counter = 0
        count = self.__class__._instance_counter
        self.__class__._instance_counter += 1
        return f"{base_name}_{count}"
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self.register_module(name, value)
        elif isinstance(value, Tensor):
            if value.requires_grad:
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)
        super().__setattr__(name, value)
        
    def register_module(self, name: str, module: Optional['Module']) -> None:
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
            module._parent = self
            module._name = name  # 子模块使用父模块给定的名称
            
    def register_parameter(self, name: str, param: Optional[Tensor]) -> None:
        if param is None:
            self._parameters.pop(name, None)
        else:
            self._parameters[name] = param
            param.name = self._full_name + '.' + name
            
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        if tensor is None:
            self._buffers.pop(name, None)
        else:
            self._buffers[name] = tensor
            tensor.name = self._full_name + '.' + name
    
    @property
    def _full_name(self) -> str:
        names = []
        module = self
        while module._parent is not None:
            names.append(module._name)
            module = module._parent
        return '.'.join(reversed(names)) if names else self._name
    
    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param
            
    def named_parameters(self, prefix: str = '', 
                        recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield (f"{prefix}{name}", param)
        if recurse:
            for module_name, module in self._modules.items():
                for p in module.named_parameters(prefix=f"{prefix}{module_name}."):
                    yield p
                    
    def children(self) -> Iterator['Module']:
        for module in self._modules.values():
            yield module
            
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        for name, module in self._modules.items():
            yield (name, module)
            
    def modules(self) -> Iterator['Module']:
        yield self
        for module in self._modules.values():
            yield from module.modules()
            
    def named_modules(self, memo: Optional[set] = None, prefix: str = ''
                     ) -> Iterator[Tuple[str, 'Module']]:
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield (prefix, self)
            for name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(memo, submodule_prefix)
                
    # def to(self, device: Union[Device, str]) -> 'Module':
    #     """移动模块到指定设备"""
    #     for param in self.parameters():
    #         param.to(device)
    #     for buf in self.buffers():
    #         buf.to(device)
    #     return self
    
    # def train(self, mode: bool = True) -> 'Module':
    #     self.training = mode
    #     for module in self.children():
    #         module.train(mode)
    #     return self
    
    # def eval(self) -> 'Module':
    #     """设置评估模式"""
    #     return self.train(False)
    
    def state_dict(self) -> Dict[str, Tensor]:
        """返回模型状态字典"""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.detach().clone()
        for name, buf in self.named_buffers():
            state[name] = buf.detach().clone()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """加载模型状态"""
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        for name, buf in self.named_buffers():
            if name in state_dict:
                buf.data.copy_(state_dict[name])
                
    def __call__(self, *args, **kwargs) -> Any:
        """允许模块像函数一样调用"""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        """前向传播（需子类实现）"""
        raise NotImplementedError("forward() not implemented")
        
    def extra_repr(self) -> str:
        """扩展信息显示（子类可重写）"""
        return ""
    
    def __repr__(self) -> str:
        """模块表示形式"""
        main_str = f"{self.__class__.__name__}({self.extra_repr()})"
        if len(self._modules):
            child_strs = []
            for name, module in self._modules.items():
                mod_str = "\n  ".join(repr(module).split("\n"))
                child_strs.append(f"({name}): {mod_str}")
            main_str += "\n  " + "\n  ".join(child_strs)
        return main_str

class Sequential(Module):
    """顺序容器模块"""
    
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
            
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x