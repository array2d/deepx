from ._datanode import DataNode
from ._opnode import OpNode
from ._controlflownode import ControlFlowNode
 
class Graph:
    # 类属性存储默认实例
    _default_graph = None
    
    
    @classmethod
    def get_default(cls):
        """获取或创建默认计算图（线程不安全）"""
        if cls._default_graph is None:
            cls._default_graph = Graph()
        return cls._default_graph
    
    @classmethod
    def set_default(cls, graph):
        """设置新的默认计算图（用于上下文管理）"""
        if not isinstance(graph, Graph):
            raise TypeError("Must be a Graph instance")
        cls._default_graph = graph

    def __init__(self,eager=True):
        self.nodes = []
        self.inputs = []
        self.var_counter = 0
        self.vector_counter = 0
        self.tensor_counter = 0
        self.control_flow_counter = 0
        self.eager=eager
    
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



# 初始化默认图
Graph._default_graph = Graph()