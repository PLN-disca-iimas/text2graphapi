import networkx as nx
import networkx
import pandas as pd
import numpy as np

class GraphTransformation():
    def __init__(self):
        pass

    # transform nx-graph to adj matrix
    def to_adjacency_matrix(graph: networkx, sparse=True) -> np:
        return nx.adjacency_matrix(graph)
    
    # transform nx-graph to adj matrix
    def to_adjacency_list(graph: networkx) -> list:
        return nx.generate_adjlist(graph)
    
    # transform nx-graph to pandas dataframe
    def to_pandas_adjacency(graph: networkx) -> pd:
        return nx.to_pandas_adjacency(graph)

    # graph trnsformations
    def transform(self, output_format: str, graph: networkx):
        print(output_format, graph)
        if output_format == 'adj_matrix':
            return self.to_adjacency_matrix(graph)
        elif output_format == 'adj_list':
            return self.to_adjacency_list(graph)
        elif output_format == 'adj_pandas':
            return self.to_pandas_adjacency(graph)
        else:
            return graph