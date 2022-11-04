import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time

from src.models.Graph import Graph
from src.Preprocessing import Preprocessing
from src.GraphTransformation import GraphTransformation

# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Cooccurence
------------
This module generate a word-coocurrence graph from raw text 
"""

class Cooccurrence(Graph):
    def __init__(self, graph_type, output_format='', apply_prep=True, window_size=1):
        """
        Co-occurrence settings, define the params to generate graph
        :param graph_type: str
        :param output_format: str
        :param apply_prep: bool
        :param window_size: int
        """
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.prep = Preprocessing()
        self.graph_trans = GraphTransformation()
        self.graph_type = graph_type
        self.output_format = output_format
        #super().__init__(graph_type, output_format)

    # normalize text
    def __text_normalize(self, text: str) -> list:
        if self.apply_prep == True:
            text = self.prep.handle_blank_spaces(text)
            text = self.prep.handle_non_ascii(text)
            text = self.prep.handle_emoticons(text)
            text = self.prep.handle_html_tags(text)
            text = self.prep.handle_stop_words(text)
            text = self.prep.to_lowercase(text)

        #preproc baseline: word_tokenize
        text = self.prep.word_tokenize(text)
        return text


    # get nodes an its attributes
    def __get_entities(self, word_tokens: list) -> list:  
        nodes = []
        word_tokens_tags = self.prep.pos_tagger(word_tokens)
        for n in word_tokens_tags:
            node = (n[0], {'pos_tag': n[1]}) # (word, {'node_attr': value})
            nodes.append(node) 
        return nodes

    # get edges an its attributes
    def __get_relations(self, word_tokens: list) -> list:
        edges = []
        d_cocc = defaultdict(int)
        for i in range(len(word_tokens)):
            word = word_tokens[i]
            next_word = word_tokens[i+1 : i+1 + self.window_size]
            for t in next_word:
                key = (word, t)
                d_cocc[key] += 1
        for key, value in d_cocc.items():
            edge = (key[0], key[1], {'freq': value})  # (word_i, word_j, {'edge_attr': value})
            edges.append(edge) 
        return edges
    

    # build nx-graph based of nodes and edges
    def __build_graph(self, nodes: list, edges: list) -> networkx:
        # pending validations
        graph = super().set_graph_type(self.graph_type)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


    # tranform raw_text to graph based on input params
    def transform(self, corpus_input_text: list) -> list:
        """
        This method transform input raw_text to graph representation
        :param corpus: list
        :return: list
        """
        logger.info("Inint text to graph tranformation")
        corpus_output_graph = []
        doc_id = 1
        avg_time = 0
        number_of_edges = 0
        number_of_nodes = 0
        for instance in corpus_input_text:
            logger.debug('--- Processing doc ', str(doc_id))
            time_init = time.time() # time init
            output_dict = {'doc_id': instance['id'], 'doc_graph': '', 'status': 'success'}
            print(instance)
            try:
                #1. text preprocessing
                prep_text = self.__text_normalize(instance['doc'])
                #2. get entities
                nodes = self.__get_entities(prep_text)
                #3. get relations
                edges = self.__get_relations(prep_text)
                #4. build graph
                graph = self.__build_graph(nodes, edges)
                number_of_edges += graph.number_of_edges()
                number_of_nodes += graph.number_of_nodes()
                #5. graph_transformation
                output_dict['doc_graph'] = self.graph_trans.transform(self.output_format, graph)
            except Exception as e:
                logger.error('Error: %s', str(e))
                output_dict['status'] = 'fail'
            finally:
                corpus_output_graph.append(output_dict)
                doc_id += 1

            time_end = time.time() - time_init # time end
            avg_time += time_end
            logger.debug("time: %s sec", str(time_end))
            
        logger.debug('Avg. time: %s', str(avg_time/doc_id))
        logger.debug('Avg. nodes: %s', str(number_of_nodes//doc_id))
        logger.debug('Avg. edges: %s', str(number_of_edges//doc_id))

        return corpus_output_graph

