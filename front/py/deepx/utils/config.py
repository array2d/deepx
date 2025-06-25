import copy,os,json
from typing import Any, Dict, List, Optional, Tuple, Union

class Config:
    """
    通用配置类，支持点号访问和递归嵌套结构。
    可从字典或对象初始化，并提供字典式的访问方法。
    """
    def __init__(self, obj: Optional[Union[Dict, object]] = None) -> None:
        """
        初始化配置对象。
        
        Args:
            obj: 字典、对象或None（默认创建空配置）。
        """
        if obj is None:
            obj = {}
        if isinstance(obj, dict):
            # 深拷贝字典以避免外部修改
            for key, value in copy.deepcopy(obj).items():
                setattr(self, key, self._process_value(value))
        else:
            # 仅复制实例属性（vars()等价于obj.__dict__）
            for key, value in vars(obj).items():
                setattr(self, key, self._process_value(value))
    
    def _process_value(self, value: Any) -> Any:
        """递归处理值，将字典转换为Config，列表/元组递归处理。"""
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [self._process_value(item) for item in value]
        elif isinstance(value, tuple):
            # 可选：将元组转换为列表以支持点号访问
            # return ConfigList([self._process_value(item) for item in value])
            return tuple(self._process_value(item) for item in value)
        else:
            return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)
    
    def __delitem__(self, key: str) -> None:
        delattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __iter__(self) -> Any:
        return iter(self.__dict__)
    
    def __repr__(self) -> str:
        return f"Config({self.__dict__})"
    
    def __str__(self) -> str:
        return str(self.to_dict())
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict:
        """将配置递归转换为字典。"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, Config) else item for item in value]
            elif isinstance(value, tuple):
                result[key] = tuple(item.to_dict() if isinstance(item, Config) else item for item in value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_file(cls, filepath: str) -> "Config":
        """
        从本地 JSON 或 YAML 文件加载配置，返回 Config 实例。
        支持 .json, .yaml, .yml 文件。
        """
        ext = os.path.splitext(filepath)[-1].lower()
        with open(filepath, "r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
            elif ext in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
        return cls(data)