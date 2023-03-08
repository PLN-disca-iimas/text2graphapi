import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from joblib import Parallel, delayed
import warnings

from text2graphapi.src.Utils import Utils
from text2graphapi.src.Preprocessing import Preprocessing
from text2graphapi.src.GraphTransformation import GraphTransformation
from text2graphapi.src import Graph

warnings.filterwarnings("ignore")

# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Cooccurrence(Graph.Graph):
    """This module generate a word-coocurrence graph from raw text 
        
        :param str graph_type: graph type to generate, default=Graph (types: Graph, DiGraph, MultiGraph, MultiDiGraph)
        :param str output_format: output format to the graph default=networkx (formats: networkx, adj_matrix, adj_list, adj_pandas)
        :param int window_size: windows size for co-occurrence, default=1
        :param int language: language for text prepocessing, default=en (lang: en, es)
        :param bool apply_prep: flag to exec text prepocessing, default=true
        :param bool parallel_exec: flag to exec tranformation in parallel, default=false
    """
    def __init__(self, graph_type, output_format='', apply_prep=True, window_size=1, parallel_exec=True, language='en'):
        """Constructor method
        """
        self.apply_prep = apply_prep
        self.parallel_exec = parallel_exec
        self.window_size = window_size
        self.prep = Preprocessing(lang=language)
        self.utils = Utils()
        self.graph_trans = GraphTransformation()
        self.graph_type = graph_type
        self.output_format = output_format
        #super().__init__(graph_type, output_format)


    # normalize text
    def _text_normalize(self, text: str) -> list:
        if self.apply_prep == True:
            text = self.prep.handle_blank_spaces(text)
            text = self.prep.handle_non_ascii(text)
            text = self.prep.handle_emoticons(text)
            text = self.prep.handle_html_tags(text)
            text = self.prep.handle_stop_words(text)
            text = self.prep.to_lowercase(text)

        #preproc baseline: word_tokenize
        #text = self.prep.word_tokenize(text)
        return text


    # get nodes an its attributes
    def _get_entities(self, text_doc: list) -> list:  
        nodes = []
        word_tokens_tags = self.prep.pos_tagger(text_doc)
        for n in word_tokens_tags:
            node = (str(n[0]), {'pos_tag': n[1]}) # (word, {'node_attr': value})
            nodes.append(node)

        logger.debug("Nodes: %s", nodes)
        return nodes


    # get edges an its attributes
    def _get_relations(self, text_doc: list) -> list:
        text_doc_tokens = self.prep.word_tokenize(text_doc)
        edges = []
        d_cocc = defaultdict(int)
        for i in range(len(text_doc_tokens)):
            word = text_doc_tokens[i]
            next_word = text_doc_tokens[i+1 : i+1 + self.window_size]
            for t in next_word:
                key = (word, t)
                d_cocc[key] += 1
        for key, value in d_cocc.items():
            edge = (key[0], key[1], {'freq': value})  # (word_i, word_j, {'edge_attr': value})
            edges.append(edge) 

        logger.debug("Edges: %s", edges)
        return edges
    

    # build nx-graph based of nodes and edges
    def _build_graph(self, nodes: list, edges: list) -> networkx:
        # pending validations
        graph = super().set_graph_type(self.graph_type)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


    def _transform_pipeline(self, text_instance: list) -> list:
        output_dict = {
            'doc_id': text_instance['id'], 
            'graph': None, 
            'number_of_edges': 0, 
            'number_of_nodes': 0, 
            'status': 'success'
        }
        try:
            #1. text preprocessing
            prep_text = self._text_normalize(text_instance['doc'])
            #2. get_entities
            nodes = self._get_entities(prep_text)
            #3. get_relations
            edges = self._get_relations(prep_text)
            #4. build graph
            graph = self._build_graph(nodes, edges)
            output_dict['number_of_edges'] += graph.number_of_edges()
            output_dict['number_of_nodes'] += graph.number_of_nodes()
            #5. graph_transformation
            output_dict['graph'] = self.graph_trans.transform(self.output_format, graph)
        except Exception as e:
            logger.error('Error: %s', str(e))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
        

    # tranform raw_text to graph based on input params
    def transform(self, corpus_texts) -> list:
        """
        This method transform input raw_text to graph representation

        :param list corpus_texts: input texts to transform into graphs 
        
        :return: list
        
        Input Example

        .. highlight:: python
        .. code-block:: python
            
            corpus_texts = [
                {  
                    "id": 1, 
                    "doc": "text_data"
                }, ...
            ]
        
        Output Example

        .. highlight:: python
        .. code-block:: python
            
            [
                {
                    "id": 1, 
                    "doc_graph": adj_matrix, 
                    'number_of_edges': 123, 
                    'number_of_nodes': 321 
                    'status': 'success'
                }, ...
            ]

            

        """
        logger.info("Init transformations: Text to Co-Ocurrence Graph")
        logger.info("Number of Text Documents: %s", len(corpus_texts))
        corpus_output_graph, delayed_func = [], []

        for input_text in corpus_texts:
            logger.debug('--- Processing doc %s ', str(input_text['id'])) 
            if self.parallel_exec == True: 
                delayed_func.append(
                    self.utils.joblib_delayed(funct=self._transform_pipeline, params=input_text) 
                )
            else:
                corpus_output_graph.append(self._transform_pipeline(input_text))

        if self.parallel_exec == True: 
            corpus_output_graph = Parallel(n_jobs=-1)(delayed_func)
        
        logger.info("Done transformations")
        return corpus_output_graph

