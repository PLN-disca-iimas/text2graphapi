import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from math import log
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import collections
import itertools
import os
import warnings


# Logging configs
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .Utils import Utils
from .Preprocessing import Preprocessing
from .GraphTransformation import GraphTransformation
from .Graph import Graph
from .configs import ENV_EXECUTION, NUM_PRINT_ITER, OUTPUT_DIR_HETERO_PATH

logger.debug('Import libraries/modules from :%s', ENV_EXECUTION)


class Heterogeneous(Graph):
    """This module generate a Heterogeneous graph from raw text 

        :param str graph_type: graph type to generate, default=Graph (types: Graph, DiGraph, MultiGraph, MultiDiGraph)
        :param str output_format: output format to the graph default=networkx (formats: networkx, adj_matrix, adj_list, adj_pandas)
        :param int window_size: windows size for pmi measure, default=20
        :param int language: language for text prepocessing, default=en (lang: en, es)
        :param bool apply_prep: flag to exec text prepocessing, default=true
        :param bool parallel_exec: flag to exec tranformation in parallel, default=false
    """
    def __init__(self, 
                graph_type, 
                output_format='', 
                window_size=20, 
                parallel_exec=False, 
                language='en', 
                apply_prep=True, 
                steps_preprocessing={},
                load_preprocessing=False
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
        self.load_preprocessing = load_preprocessing


    # build vocab
    # get word set and word frequency
    def __build_vocab(self, corpus):
        word_set = set()
        doc_words_list = []
        for i in range(len(corpus)):
            doc_words = corpus[i]['doc']
            words = self.prep.word_tokenize(doc_words)
            doc_words_list.append({'doc': i, 'words': words})
            for word in words:
                word_set.add(word)
        return doc_words_list, list(word_set)


    # get windows
    # words windows based on window_size param -> [[w1,w2,...], [w3,w4,...], ...]
    def __get_windows(self, doc_words_list, window_size):
        word_window_freq = defaultdict(int)  
        word_pair_count = defaultdict(int)
        len_doc_words_list = len(doc_words_list)
        len_windows = 0

        for i, doc in enumerate(doc_words_list):
            windows = []
            doc_words = doc['words']
            length = len(doc_words)

            if (i+1) == NUM_PRINT_ITER * ((i+1)//NUM_PRINT_ITER):
                logger.debug("\t Iter %s out of %s", str(i+1), str(len_doc_words_list))

            if length <= window_size:
                windows.append(doc_words)
            else:
                for j in range(length - window_size + 1):
                    window = doc_words[j: j + window_size]
                    windows.append(list(set(window)))

            for window in windows:
                for word in window:
                    word_window_freq[word] += 1
                for word_pair in itertools.combinations(window, 2):
                    word_pair_count[word_pair] += 1    
            len_windows += len(windows)

        return word_window_freq, word_pair_count, len_windows
    

    # get pmi measure
    # pmi for pair of word,word -> {'w1,w2': pmi, 'w5,w6': pmi,  ...}
    def __get_pmi(self, doc_words_list, window_size):
        logger.debug('\t Get PMI measure')
        word_window_freq, word_pair_count, len_windows = self.__get_windows(doc_words_list, window_size)
        word_to_word_pmi = []
        for word_pair, count in word_pair_count.items():
            word_freq_i = word_window_freq[word_pair[0]]
            word_freq_j = word_window_freq[word_pair[1]]
            pmi = log((1.0 * count / len_windows) / (1.0 * word_freq_i * word_freq_j/(len_windows * len_windows)))
            if pmi <= 0:
                continue
            word_to_word_pmi.append((word_pair[0], word_pair[1], {'pmi': round(pmi, 2)}))
        return word_to_word_pmi      
    

    # get tfidf meausure
    # tfid for relation doc,word -> {'doc1,word1': tfidf, 'doc1,word2': tfidf,  ...}
    def __get_tfidf(self, corpus_docs_list, vocab):
        logger.debug('\t Get TF-IDF measure')
        vectorizer = TfidfVectorizer(vocabulary=vocab, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False, lowercase=False, tokenizer=None)
        tfidf = vectorizer.fit_transform(corpus_docs_list)
        words_docs_tfids = []
        len_tfidf = tfidf.shape[0]

        for ind, row in enumerate(tfidf):
            if (ind+1) == NUM_PRINT_ITER * ((ind+1)//NUM_PRINT_ITER):
                logger.debug("\t Iter %s out of %s", str(ind+1), str(len_tfidf))
            for col_ind, value in zip(row.indices, row.data):
                edge = ('D-' + str(ind+1), vocab[col_ind], {'tfidf': round(value, 2)})
                words_docs_tfids.append(edge)
        return words_docs_tfids


    def _text_normalize(self, text: str) -> list:
        """This module generate a word-coocurrence graph from raw text 

            :param str text: texto to normalize 
        """
        prep_text = self.prep.prepocessing_pipeline(text)
        return prep_text


    def _text_normalize_2(self, text_docs: dict) -> list:
        corpus_docs_list = []
        doc_words_list = []
        vocab = set()
        text_docs_tuple = []
        for i in range(len(text_docs)):
            prep_text = self.prep.prepocessing_pipeline(text_docs[i]['doc'])
            text_docs_tuple.append((prep_text, {'id': text_docs[i]['id']}))
            if (i+1) == NUM_PRINT_ITER * ((i+1)//NUM_PRINT_ITER):
                logger.info("\t Iter %s out of %s", str(i+1), str(len(text_docs)))
 
        doc_nlp = self.prep.nlp_pipeline(text_docs_tuple)
        for doc, context in doc_nlp:
            #doc_tokens = [str(token) for token in doc]
            doc_tokens = [str(token.lemma_) for token in doc]
            #print(doc.text, context['id'])
            corpus_docs_list.append(str(doc.text))
            doc_words_list.append({'doc': context['id'], 'words': doc_tokens})
            vocab.update(set(doc_tokens))
            '''res_doc = {
                'id': context['id'],
                'doc': doc.text,
                'doc_tokenize': [str(token) for token in doc],
            }
            res.append(res_doc)'''
        return corpus_docs_list, doc_words_list, list(vocab)
    

    # get nodes an its attributes
    def __get_entities(self, doc_words_list: list) -> list:  
        nodes = []
        for d in doc_words_list:
            node_doc =  ('D-' + str(d['doc']), {})
            nodes.append(node_doc)
            for word in d['words']:
                node_word = (str(word), {})
                nodes.append(node_word)
        return nodes


   # get edges an its attributes
    def __get_relations(self, corpus_docs_list, doc_words_list, vocab) -> list:  
        edges = []
        #tfidf
        word_to_doc_tfidf = self.__get_tfidf(corpus_docs_list, vocab)
        edges.extend(word_to_doc_tfidf)
        #pmi
        word_to_word_pmi = self.__get_pmi(doc_words_list, self.window_size)
        edges.extend(word_to_word_pmi)
        # return relations/edges
        return edges


    # build nx-graph based of nodes and edges
    def __build_graph(self, nodes:list, edges: list) -> networkx:
        # pending validations
        graph = super().set_graph_type(self.graph_type)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


    def __transform_pipeline(self, corpus_docs: list) -> list:
        output_dict = {
            'doc_id': 1, 
            'graph': None, 
            'number_of_edges': 0, 
            'number_of_nodes': 0, 
            'status': 'success'
        }
        try:

            #1. text preprocessing
            logger.debug("1. text preprocessing")
            corpus_docs_list = []
            doc_words_list = []
            len_corpus_docs = len(corpus_docs)
            vocab = set()
            delayed_func = []
            prep_docs = []

            logger.debug("Preprocessing")
            for doc_data in corpus_docs:
                if self.apply_prep == True:
                    doc_data['doc'] = self._text_normalize(doc_data['doc'])
                prep_docs.append((doc_data['doc'], {'id': doc_data['id']}))

            logger.debug("Spacy nlp_pipeline")
            docs = self.prep.nlp_pipeline(prep_docs)

            logger.debug("Get Vocab...")
            for doc, context in docs:
                #doc_tokens = [str(token) for token in doc]
                doc_tokens = [str(token.lemma_) for token in doc]
                #print(doc.text, context['id'])
                corpus_docs_list.append(str(" ".join(doc_tokens)))
                doc_words_list.append({'doc': context['id'], 'words': doc_tokens})
                vocab.update(set(doc_tokens))
            
            #2. get node/entities
            logger.debug("3. Get node/entities")
            nodes = self.__get_entities(doc_words_list)

            #3. get edges/relations
            logger.debug("4. Get edges/relations")
            edges = self.__get_relations(corpus_docs_list, doc_words_list, list(vocab))
            
            #4. build graph
            logger.debug("5. Build graph")
            graph = self.__build_graph(nodes, edges)
            output_dict['number_of_edges'] = graph.number_of_edges()
            output_dict['number_of_nodes'] = graph.number_of_nodes()

            #5. graph_transformation
            logger.debug("6. Transform output graph")
            output_dict['graph'] = self.graph_trans.transform(self.output_format, graph)
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.error('Error Detail: %s', str(traceback.format_exc()))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
        

    def transform(self, corpus_docs: list) -> list:
        logger.info("Init transformations: Text to Heterogeneous Graph")
        logger.info("Transforming %s text documents...", len(corpus_docs))

        corpus_output_graph = [self.__transform_pipeline(corpus_docs)]
        #self.utils.save_data(data=corpus_output_graph, path=OUTPUT_DIR_HETERO_PATH, file_name='corpus_graph', compress=1)

        logger.info("Done transformations")
        return corpus_output_graph
    



