from .node import Node
from .nodetype import NodeType 
 
class DataNode(Node):
    def __init__(self, name=None, type=None, data=None):
        super().__init__(name=name, ntype=NodeType.DATA)
        self._data = data
        self._type=type
    @property
    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data
 
 
