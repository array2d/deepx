from ._datanode import DataNode
from ._opnode import OpNode
from ._controlflownode import ControlFlowNode


class Graph:
    def __init__(self,eager=True):
        self.nodes = []
        self.inputs = []
        self.var_counter = 0
        self.vector_counter = 0
        self.tensor_counter = 0
        self.control_flow_counter = 0
        self.eager=eager
        self.tracegraph=True
 

    @property
    def tracegraph(self):
        return self._tracegraph
    @tracegraph.setter
    def tracegraph(self,value:bool):
        self._tracegraph=value

    @property
    def eager(self):
        return self._eager
    @eager.setter
    def eager(self,value):
        self._eager=value

    def add_var(self, name,data,inputs=[]):
        self.var_counter += 1
        if name == "" or name is None:
            name = f"var_{self.var_counter}"
        node=DataNode(name, "var", data)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
    
    def add_vector(self, name,data,inputs=[]):
        self.vector_counter += 1
        if name == "" or name is None:
            name = f"vector_{self.vector_counter}"
        node=DataNode(name, "vector", data)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
    
    def add_tensor(self, name,t,inputs=[]):
        self.tensor_counter += 1
        if name == "" or name is None:
            name = f"tensor_{self.tensor_counter}"
        node=DataNode(name, "tensor", t)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node


    def add_op(self,name,inputs=[]):
        node=OpNode(name)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
    def add_control_flow(self,name,inputs=[]):
        self.control_flow_counter += 1
        if name == "":
            name = f"control_flow_{self.control_flow_counter}"
        node=ControlFlowNode(name)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
def graph_method(f):
    setattr(Graph, f.__name__, f)
    return f

 