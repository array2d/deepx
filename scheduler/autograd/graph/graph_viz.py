import graphviz
from .graph import Graph,graph_method
from ._datanode import DataNode
from ._opnode import OpNode


@graph_method
def to_dot(self)->graphviz.Digraph:
        """生成DOT格式的计算图可视化""" 
        dot = graphviz.Digraph(comment='Computational Graph')
        dot.attr(rankdir='TB')  # 从上到下的布局
        
        # 设置全局节点属性
        dot.attr('node', shape='record')
        
        # 添加节点
        for node in self.nodes:
            attrs = {
                'fontname': 'Sans-Serif',
                'labeljust': 'l'
            }
            
            if isinstance(node, DataNode):
                label=f'{node.name}\n'
                match(node._type):
                    case "var":
                        label += f"{node.data}"
                        attrs.update({
                            'shape': 'box',
                            'color': 'orange',
                            'style': 'filled',
                            'fillcolor': 'moccasin'
                        })
                    case "vector":
                        label += f"{node.data}"
                        attrs.update({
                            'shape': 'box',
                            'color': 'darkseagreen',
                            'style': 'filled',
                            'fillcolor': 'honeydew'
                        })
                    case "tensor":
                        label += f"{node.data.shape if node.data else ''}"
                        attrs.update({
                            'shape': 'box',
                            'color': 'skyblue',
                            'style': 'filled',
                            'fillcolor': 'aliceblue'
                        })
               
                
            elif isinstance(node, OpNode):
                # 操作节点：突出显示操作类型
                label = node.name
                attrs.update({
                    'shape': 'box',
                    'style': 'filled',
                    'fillcolor': 'lightgray',
                    'color': 'darkslategray',
                    'fontname': 'Courier Bold'
                })
            # 添加节点
            dot.node(str(id(node)), label, **attrs)
        
        # 添加边连接
        for node in self.nodes:
            for input_node in node.inputs:
                dot.edge(
                    str(id(input_node)), 
                    str(id(node)),
                    color='gray40',
                    penwidth='1.2',
                    arrowsize='0.8'
                )
        
        return dot 