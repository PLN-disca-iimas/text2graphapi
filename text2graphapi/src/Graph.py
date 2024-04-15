import matplotlib.pyplot as plt
import networkx as nx
import networkx
import random

class Graph(object): 
    """
    Graph general settings

    :param graph_type: str
    :param output_format: str
    """
    def __init__(self, graph_type='Graph', output_format= 'adj_matrix'):
        """Constructor method
        """
        self.output_format = output_format
        self.graph = self.set_graph_type(graph_type)


    def set_graph_type(self, graph_type: str) -> networkx:
        '''
        '''
        graph = None
        if graph_type == 'MultiDiGraph':
            graph = nx.MultiDiGraph()
        elif graph_type == 'MultiGraph':
            graph = nx.MultiGraph()
        elif graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        return graph


    def plot(self, graph: nx.DiGraph, output_path: str):
        """
            This method allow to plot a networkx graph
            
            :param networkx graph: graph to plot
            :param output_path: Path for image output 
            :returns: graph
            
            :rtype: none
        """
        nodes_colors = [random.randint(0, 100) / 1000 for node in graph.nodes()]
        colors_options = ["r","k","b"]

        edges_colors = [random.choice(colors_options) for edge in graph.edges()]


        pos = nx.spring_layout(graph, k=1, iterations=20)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('tab20'), node_color=nodes_colors, node_size=100)
        nx.draw_networkx_labels(graph, pos, font_size=5)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), arrows=True, arrowsize=5, edge_color=edges_colors)
        plt.savefig(output_path, dpi=300)