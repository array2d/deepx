from .nodetype import NodeType

class Node:
    def __init__(self, 
                ntype:NodeType=None,
                name:str=None,
                graph=None,
                ):
        from .graph import Graph
        if graph == None:
            self._graph = Graph.get_default()
        else:
            self._graph = graph
        self._module = None
        self._ntype = ntype
        self._name = name
        self._inputs = []

    @property
    def ntype(self):
        return self._ntype
    
    @property
    def graph(self):
        return self._graph
    
    @property
    def name(self):
        return self._name
    
    def rename(self,name:str):
        self._name = name

    
    @property
    def fullname(self):
        if self._module is None:
            return self._name
        else:
            return f"{self._module.full_name}.{self._name}"
    
    def set_module(self,module):
        from deepx.nn.modules import Module
        if isinstance(module,Module):
            self._module = module
        else:
            raise ValueError("module must be a Module")

    @property
    def module(self):
        return self._module

    @property
    def inputs(self):
        return self._inputs

    
    def add_input(self, input_node):
        self._inputs.append(input_node)