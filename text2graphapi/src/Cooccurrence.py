import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from joblib import Parallel, delayed
import warnings

# Configs
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from .Utils import Utils
from .Preprocessing import Preprocessing
from .GraphTransformation import GraphTransformation
from .Graph import Graph
from .configs import ENV_EXECUTION, NUM_PRINT_ITER

logger.debug('Import libraries/modules from :%s', ENV_EXECUTION)

class Cooccurrence(Graph):
    """This module generate a word-coocurrence graph from raw text 
        
        :param str graph_type: graph type to generate, default=Graph (types: Graph, DiGraph, MultiGraph, MultiDiGraph)
        :param str output_format: output format to the graph default=networkx (formats: networkx, adj_matrix, adj_list, adj_pandas)
        :param int window_size: windows size for co-occurrence, default=1
        :param int language: language for text prepocessing, default=en (lang: en, es, fr)
        :param bool apply_prep: flag to exec text prepocessing, default=true
        :param bool parallel_exec: flag to exec tranformation in parallel, default=false
    """
    def __init__(self, 
                graph_type, 
                output_format='', 
                apply_prep=True, 
                window_size=1,
                parallel_exec=False, 
                language='en', 
                steps_preprocessing={}
            ):
        """Constructor method
        """
        self.apply_prep = apply_prep
        self.parallel_exec = parallel_exec
        self.window_size = window_size
        self.prep = Preprocessing(lang=language, steps_preprocessing=steps_preprocessing)
        self.utils = Utils()
        self.graph_trans = GraphTransformation()
        self.graph_type = graph_type
        self.output_format = output_format
        #super().__init__(graph_type, output_format)


    # normalize text
    def _text_normalize(self, text: str) -> list:
        """This module generate a word-coocurrence graph from raw text 

            :param str text: texto to normalize 
        """
        prep_text = self.prep.prepocessing_pipeline(text)
        return prep_text


    # get nodes an its attributes
    def _get_entities(self, doc_instance) -> list:  
        nodes = []
        #word_tokens_tags = self.prep.pos_tagger(text_doc)
        #ext_doc_id = "preproc_text_" + str(doc2['id'])
        #print('get_entities: ')

        #print(ext_doc_id, doc['doc'].get_extension(ext_doc_id))
        for token in doc_instance:
            #print(token.text, token.lemma_, token.pos_, token.is_stop)
            #if token.lower() not in doc.get_extension("preproc_text"):
            #    continue
            node = (str(token.lemma_), {'pos_tag': token.pos_}) # (word, {'node_attr': value})
            nodes.append(node)

        logger.debug("Nodes: %s", nodes)
        return nodes


    # get edges an its attributes
    def _get_relations(self, doc) -> list:
        #print('_get_relations')
        
        text_doc_tokens = [token.lemma_ for token in doc]
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


    def _transform_pipeline(self, doc_instance: tuple) -> list:
        output_dict = {
            'doc_id': doc_instance['id'], 
            'graph': None, 
            'number_of_edges': 0, 
            'number_of_nodes': 0, 
            'status': 'success'
        }
        try:
            #1. text preprocessing
            #prep_text = self._text_normalize(doc_instance['doc'])
            
            #2. get_entities
            nodes = self._get_entities(doc_instance['doc'])
            
            #3. get_relations
            edges = self._get_relations(doc_instance['doc'])
            
            #4. build graph
            graph = self._build_graph(nodes, edges)
            output_dict['number_of_edges'] += graph.number_of_edges()
            output_dict['number_of_nodes'] += graph.number_of_nodes()
            
            #5. graph_transformation
            output_dict['graph'] = self.graph_trans.transform(self.output_format, graph)
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.debug('Error Detail: %s', str(traceback.format_exc()))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
        

    # tranform raw_text to graph based on input params
    def transform(self, corpus_texts) -> list:
        """
        This method transform input raw_text to graph representation

        :param list corpus_texts: input texts to transform into graphs 
        
        :return: list
        
        """
        logger.info("Init transformations: Text to Co-Ocurrence Graph")
        logger.info("Transforming %s text documents...", len(corpus_texts))
        prep_docs, corpus_output_graph, delayed_func = [], [], []

        logger.debug("Preprocessing")
        for doc_data in corpus_texts:
            if self.apply_prep == True:
                doc_data['doc'] = self._text_normalize(doc_data['doc'])
            prep_docs.append((doc_data['doc'], {'id': doc_data['id']}))
        docs = self.prep.nlp_pipeline(prep_docs)

        logger.debug("Transform_pipeline")
        for doc, context in list(docs):
            corpus_output_graph.append(self._transform_pipeline({'id': context['id'], 'doc': doc}))
    
        logger.info("Done transformations")
        return corpus_output_graph
        


'''
if self.parallel_exec == True: 
    for input_text in corpus_texts:
        logger.debug('--- Processing doc %s ', str(input_text['id'])) 
        delayed_func.append(
            self.utils.joblib_delayed(funct=self._transform_pipeline, params=input_text) 
        )
    corpus_output_graph = self.utils.joblib_parallel(delayed_func, process_name='transform_cooocur_graph')
else:
    for input_text in corpus_texts:
        logger.debug('--- Processing doc %s ', str(input_text['id'])) 
        corpus_output_graph.append(self._transform_pipeline(input_text))
'''