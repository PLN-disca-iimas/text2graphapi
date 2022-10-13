import networkx as nx
import networkx
import pandas as pd
import numpy as np

class GraphTransformation(object):
    def __init__():
        pass

    # transform nx-graph to adj matrix
    def to_adjacency_matrix(graph: networkx, sparse=True) -> np:
        pass
    
    # transform nx-graph to adj matrix
    def to_adjacency_list(graph: networkx) -> list:
        pass
    
    # transform nx-graph to pandas dataframe
    def to_pandas_adjacency(graph: networkx) -> pd:
        pass

    # graph trnsformations
    def transform(self, output_format, graph: networkx):
        if output_format == 'adj_matrix':
            return self.to_adjacency_matrix(graph)
        elif output_format == 'adj_list':
            return self.to_adjacency_list(graph)
        elif output_format == 'adj_pandas':
            return self.to_pandas_adjacency(graph)
        else:
            return graph