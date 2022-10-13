import networkx as nx
import networkx
import Graph
from src import Preprocessing
from src import GraphTransformation

"""
Cooccurence
------------
This module generate a word-coocurrence graph from raw text 
"""

class Cooccurrence(Graph):
    def __init__(self, graph_type='DiGraph', apply_prep=True, window_size=1, output_format=None):
        """
        Co-occurrence settings, define the params to generate graph
        :param graph_type: str
        :param apply_prep: bool
        :param window_size: int
        :param output_format: str
        """
        self.graph_type = graph_type
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.output_format = output_format
        self.graph_corpus, self.nodes, self.edges = [], []


    # normalize text
    def __text_normalize(self, text: str) -> str:
        prep = Preprocessing()
        if self.apply_prep == True:
            ...

        #preproc baseline
        text = prep.word_tokenize(text)
        ...
        return text


    # get nodes an its attributes
    # structure: (node_word, {'node_attr': value})
    def __get_entities(self, text: str) -> list:
        pass


    # get edges an its attributes
    # structure: (word_i, word_j, {'edge_attr': value})
    def __get_relations(self, text: str) -> list:
        pass
    

    # build nx-graph based of nodes and edges
    def __build_graph(self, graph_type: str, nodes: list, edges: list) -> networkx:
        pass

    # tranform raw_text to graph based on input params
    def transform(self, corpus: list):
        """
        This method transform input raw_text to graph representation
        :param corpus: list
        :return: list
        """
        for text in corpus:
            #1. text preprocessing
            prep_text = self.__text_normalize(text)
            #2. get entities
            self.nodes = self.__get_entities(prep_text)
            #3. get relations
            self.edges = self.__get_relations(text)
            #4. build graph
            graph = self.__build_graph(self.graph_type, self.nodes, self.edges)
            #5. graph_transformation
            gr_t = GraphTransformation()
            trans_graph = gr_t.transform(self.output_format, graph)

            self.graph_corpus.append(trans_graph)

        return self.graph_corpus

 