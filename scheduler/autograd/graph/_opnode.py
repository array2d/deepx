from .node import Node
from .nodetype import NodeType
 
class OpNode(Node):
    def __init__(self, name: str):
        super().__init__(name=name, ntype=NodeType.OP)
