import matplotlib.pyplot as plt
import networkx as nx
import networkx

class Graph():
    def __init__(self, graph_type='Graph', output_format= 'adj_matrix'):
        """
        Graph general settings
        :param graph_type: str
        :param output_format: str
        """
        self.output_format = output_format
        self.graph = self.__set_graph_type(graph_type)

    def set_graph_type(self, graph_type: str) -> networkx:
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

    def plot_graph(self, graph: networkx):
        pass