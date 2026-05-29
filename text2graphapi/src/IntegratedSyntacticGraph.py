import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import warnings

# Configs
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#logger.debug('Import libraries/modules from: %s', ENV_EXECUTION)
from Utils import Utils
from text2graphapi.src.Preprocessing import Preprocessing
from GraphTransformation import GraphTransformation
from Graph import Graph
from configs import ENV_EXECUTION

logger.debug('Import libraries/modules from :%s', ENV_EXECUTION)


class ISG(Graph):
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
        self.int_synt_graph = nx.DiGraph()
        self.utils = Utils()
        self.prep = Preprocessing(lang=language, steps_preprocessing=steps_preprocessing)
        self.graph_trans = GraphTransformation()

    # normalize text
    def _text_normalize(self, text: str) -> list:
        if self.apply_prep == True:
            text = self.prep.preprocessing_pipeline(text)
        return text

    def _set_synonyms(doc: dict, word_synonnyms: list):
        if len(word_synonnyms):
            return [] 
        else:
            for syn in doc['token_synonyms']:
                if str(doc['token_lemma']) == str(syn):
                    continue
                else:
                    return [(f"{syn}_{doc['token_pos']}",{'pos_tag': doc['token_pos']})]
            return []
        
    # get nodes an its attributes
    def _get_entities(self, text_doc: list) -> list:  
        nodes = [('ROOT_0', {'pos_tag': 'ROOT_0'})]
        for d in text_doc:
            node = [(f"{d['token_lemma']}_{d['token_pos']}",{'pos_tag': d['token_pos']})]
            if len(d['token_synonyms']) and True:
                node += [(f"{d['token_synonyms'][0]}_{d['token_pos']}",{'pos_tag': d['token_pos']})]
            nodes.extend(node)
        return nodes

    def _get_relations(self, text_doc: list) -> list:
        edges = []
        
        for d in text_doc:
            edge_attr = {'gramm_relation': d['token_dependency']}
            
            # Palabra principal
            main_word = f"{d['token_lemma']}_{d['token_pos']}"
            
            # Sinónimo de la palabra actual (si existe)
            current_synonym = None
            if len(d.get('token_synonyms', [])) > 0:
                current_synonym = f"{d['token_synonyms'][0]}_{d['token_pos']}"
            
            if d['is_root_token'] == True:
                # 1. Conectar ROOT con palabra principal
                edges.append(('ROOT_0', main_word, edge_attr))
                
                # 2. Conectar ROOT con sinónimo (mismo destino que palabra principal)
                if current_synonym:
                    edges.append(('ROOT_0', current_synonym, edge_attr))
            
            else:
                # Palabra cabeza (la que gobierna sintácticamente a la actual)
                head_word = f"{d['token_head_lemma']}_{d['token_head_pos']}"
                
                # Sinónimo de la cabeza (si existe)
                head_synonym = None
                if len(d.get('token_head_synonyms', [])) > 0:
                    head_synonym = f"{d['token_head_synonyms'][0]}_{d['token_head_pos']}"
                
                # 1. Conectar cabeza principal → palabra principal
                edges.append((head_word, main_word, edge_attr))
                
                # 2. Conectar cabeza principal → sinónimo (camino alternativo)
                if current_synonym:
                    edges.append((head_word, current_synonym, edge_attr))
                
                # 3. Conectar sinónimo de cabeza → palabra principal
                if head_synonym:
                    edges.append((head_synonym, main_word, edge_attr))
                
                # 4. Conectar sinónimo de cabeza → sinónimo (camino alternativo completo)
                if head_synonym and current_synonym:
                    edges.append((head_synonym, current_synonym, edge_attr))
        
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
        #self._get_frequency_weight(int_synt_graph)
        return int_synt_graph
    
    # merge/add graph to ISG
    def _add_graph_to_ISG(self, isg_graph: networkx, graph: networkx) -> networkx:
        int_synt_graph = nx.compose(isg_graph, graph)
        return int_synt_graph
    
    def _add_graph_to_ISG_V2(self, graph: networkx) -> networkx:
        self.int_synt_graph.add_edges_from(graph.edges(data=True))
        self.int_synt_graph.add_nodes_from(graph.nodes(data=True))
    

    def _transform_pipeline(self, docs: tuple) -> list:
        int_synt_graph = nx.DiGraph()
        for doc, context in list(docs):
            doc_instance = {'id': context['id'], 'doc': doc}
            try:     
                #3. get_entities
                nodes = self._get_entities(doc_instance['doc']._.multilevel_lang_info)
                
                #4. get_relations
                edges = self._get_relations(doc_instance['doc']._.multilevel_lang_info)
                
                #5. build graph
                graph = self._build_graph(nodes, edges)
                
                #6. merge/add grapg into ISG
                int_synt_graph.add_edges_from(graph.edges(data=True))
                int_synt_graph.add_nodes_from(graph.nodes(data=True))
            except Exception as e:
                logger.error('Error: %s', str(e))
                logger.error('Error Detail: %s', str(traceback.format_exc()))
        
        self._get_frequency_weight(int_synt_graph)
        return int_synt_graph
        

    # tranform raw_text to graph based on input params
    def transform(self, corpus_texts) -> list:
        """
        This method transform input raw_text to graph representation

        :param list corpus_texts: input texts to transform into graphs 
        
        :return: list
        
        """
        logger.info("Init transformations: Text to Integrated Syntactic Graphs")
        logger.info("Transforming %s text documents...", len(corpus_texts))
        corpus_output_graph, delayed_func, delayed_func_2 = [], [], []
        int_synt_graph = nx.DiGraph()

        #1. text preprocessing
        logger.debug("Preprocessing")
        prep_docs = []
        word_doc_cnt = 0
        doc_cnt = 0
        for doc_data in corpus_texts:
            word_doc_cnt += len(self.prep.word_tokenize(doc_data['doc']))
            doc_cnt += 1 
            if self.apply_prep == True:
                doc_data['doc'] = self._text_normalize(doc_data['doc'])
            prep_docs.append((
                    doc_data['doc'], 
                    {'id': doc_data['id']}
                ))
        

        #2. get multilevel lang features from text documents (lexical, morpholocial, syntactic)
        logger.debug("Spacy nlp_pipeline")
        docs = self.prep.nlp_pipeline(prep_docs, params = {'get_multilevel_lang_features': True})

        logger.debug("Transform_pipeline")
        int_synt_graph = self._transform_pipeline(docs)

        logger.info("Done transformations")
        output_dict = {
            'doc_id': 1, 
            'graph': int_synt_graph,
            'number_of_edges': int_synt_graph.number_of_edges(), 
            'number_of_nodes': int_synt_graph.number_of_nodes(), 
            'status': 'success'
        }
        return [output_dict]
    
