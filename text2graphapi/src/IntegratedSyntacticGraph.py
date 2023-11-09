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
TEST_API_FROM = 'LOCAL' #posible values: LOCAL, PYPI
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.debug('Import libraries/modules from :%s', TEST_API_FROM)
if TEST_API_FROM == 'PYPI':
    from text2graphapi.src.Utils import Utils
    from text2graphapi.src.Preprocessing import Preprocessing
    from text2graphapi.src.GraphTransformation import GraphTransformation
    from text2graphapi.src import Graph
    from text2graphapi.src import configs
else:
    from src.Utils import Utils
    from src.Preprocessing import Preprocessing
    from src.GraphTransformation import GraphTransformation
    from src import Graph
    from src import configs


class ISG(Graph.Graph):
    """This module generate a word-coocurrence graph from raw text 
        
        :param str graph_type: graph type to generate, default=DiGraph (types: Graph, DiGraph, MultiGraph, MultiDiGraph)
        :param str output_format: output format to the graph default=networkx (formats: networkx, adj_matrix, adj_list, adj_pandas)
        :param int language: language for text prepocessing, default=en (lang: en, es, fr)
        :param bool apply_prep: flag to exec text prepocessing, default=true
        :param bool parallel_exec: flag to exec tranformation in parallel, default=false
    """
    def __init__(self, 
                graph_type, 
                output_format='', 
                apply_prep=True, 
                parallel_exec=False, 
                language='en', 
                steps_preprocessing={}
            ):
        """Constructor method
        """
        self.apply_prep = apply_prep
        self.parallel_exec = parallel_exec
        self.graph_type = graph_type
        self.output_format = output_format
        self.utils = Utils()
        self.prep = Preprocessing(lang=language, steps_preprocessing=steps_preprocessing)
        self.graph_trans = GraphTransformation()

    # normalize text
    def _text_normalize(self, text: str) -> list:
        if self.apply_prep == True:
            text = self.prep.prepocessing_pipeline(text)
            text = self.prep.handle_blank_spaces(text)
        return text

    # get nodes an its attributes
    def _get_entities(self, text_doc: list) -> list:  
        nodes = [('ROOT_0', {'pos_tag': 'ROOT_0'})]
        for d in text_doc:
            node = (
            f"{d['token']}_{d['token_pos']}",
                {'pos_tag': d['token_pos']}
            )
            nodes.append(node)
        return nodes

    # get edges an its attributes
    def _get_relations(self, text_doc: list) -> list:
        # code here, edge structure: (word_i, word_j, {'edge_attr': value})
        edges = []
        for d in text_doc:
            edge_attr = {'gramm_relation': d['token_dependency']}
            if d['is_root_token'] == True:
                edge = (
                d['token'] + '_' + d['token_pos'],
                'ROOT_0',
                edge_attr
                )
            else:
                edge = (
                f"{d['token_head']}_{d['token_head_pos']}",
                f"{d['token']}_{d['token_pos']}",
                edge_attr
                )
            edges.append(edge)
        return edges
    
    # build nx-graph based of nodes and edges
    def _build_graph(self, nodes: list, edges: list) -> networkx:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph
        # code here
    
    # get syntactic frequencies & Edge frequencies
    def _get_frequency_weight(self, graph: nx.DiGraph):
        freq_dict = defaultdict(int)    
        for edge in graph.edges(data=True):
            freq_dict[edge[2]['gramm_relation']] += 1

        for edge in graph.edges(data=True):
            edge[2]['gramm_relation'] = f'{edge[2]["gramm_relation"]}_{freq_dict[edge[2]["gramm_relation"]]+graph.number_of_edges(edge[0], edge[1])}'
        
    # merge all graph to obtain the final ISG
    def _build_ISG_graph(self, graphs: list) -> networkx:
        int_synt_graph = nx.DiGraph()
        for graph in graphs:
            int_synt_graph = nx.compose(int_synt_graph, graph)
        self._get_frequency_weight(int_synt_graph)

        return int_synt_graph
        # 1. Compose/Merge all networkx graph
        # 2. Get syntactic frequencies & Edge frequencies
        # 3. Assign weight to edges

    def _transform_pipeline(self, text_instance: list) -> list:
        try:
            #1. text preprocessing
            prep_text = self._text_normalize(text_instance['doc'])
            #2. get multilevel lang features from text documents (lexical, morpholocial, syntactic)
            multi_lang_feat_doc = self.prep.get_multilevel_lang_features(prep_text)
            #3. get_entities
            nodes = self._get_entities(multi_lang_feat_doc)
            #4. get_relations
            edges = self._get_relations(multi_lang_feat_doc)
            #5. build graph
            graph = self._build_graph(nodes, edges)
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.debug('Error Detail: %s', str(traceback.format_exc()))
        finally:
            return graph
        

    # tranform raw_text to graph based on input params
    def transform(self, corpus_texts) -> list:
        """
        This method transform input raw_text to graph representation

        :param list corpus_texts: input texts to transform into graphs 
        
        :return: list
        
        """
        logger.info("Init transformations: Text to Integrated Syntactic Graphs")
        logger.info("Transforming %s text documents...", len(corpus_texts))
        corpus_output_graph, delayed_func = [], []

        if self.parallel_exec == True: 
            for input_text in corpus_texts:
                logger.debug('--- Processing doc %s ', str(input_text['id'])) 
                delayed_func.append(
                    self.utils.joblib_delayed(funct=self._transform_pipeline, params=input_text) 
                )
            corpus_output_graph = self.utils.joblib_parallel(delayed_func, process_name='transform_ISG_graphs')
        else:
            for input_text in corpus_texts:
                logger.debug('--- Processing doc %s ', str(input_text['id'])) 
                corpus_output_graph.append(self._transform_pipeline(input_text))
        
        logger.info("Done transformations")
        isg = self._build_ISG_graph(corpus_output_graph)
        return isg
    

