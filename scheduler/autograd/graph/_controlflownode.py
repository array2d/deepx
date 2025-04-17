from .node import Node
from .nodetype import NodeType


class ControlFlowNode(Node):
    def __init__(self, name=None):
        super().__init__(name="control_flow", ntype=NodeType.CONTROL_FLOW)

