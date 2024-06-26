import matplotlib.pyplot as plt
import networkx as nx
import networkx

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


    def plot(self, graph: nx.DiGraph, output_path: str, options: dict = {}):
        """
            This method allow to plot a networkx graph
            
            :param networkx graph: graph to plot
            :param output_path: Path for image output 
            :returns: graph
            
            :rtype: none
        """
        graph._node = dict(sorted(graph._node.items()))
        node_labels = options.get("nodes_labels", {})
        edge_labels = options.get("edge_labels", {})
        graph = nx.relabel_nodes(graph, node_labels)
        pos = nx.spring_layout(graph, k=1, iterations=20)
        plt.cla()

        nx.draw_networkx_nodes(
            graph, 
            pos, 
            cmap=plt.get_cmap('tab20'), 
            **options.get("nodes_options", {}) 
        )
        nx.draw_networkx_labels(
            graph, 
            pos,
            **options.get("nodes_labels_options", {}) 
        )
        nx.draw_networkx_edges(
            graph, 
            pos,
            edgelist=graph.edges(), 
            **options.get("edges_options", {})
        )
        nx.draw_networkx_edge_labels(
            graph, 
            pos,
            edge_labels=edge_labels,
            **options.get("edges_labels_options", {})
        )
        plt.savefig(output_path, dpi=300)