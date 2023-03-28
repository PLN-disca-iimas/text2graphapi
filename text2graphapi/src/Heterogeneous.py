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

from src.Utils import Utils
from src.Preprocessing import Preprocessing
from src.GraphTransformation import GraphTransformation
from src import Graph
from src import configs


# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Heterogeneous(Graph.Graph):
    """This module generate a Heterogeneous graph from raw text 
        :param str graph_type: graph type to generate, default=Graph (types: Graph, DiGraph, MultiGraph, MultiDiGraph)
        :param str output_format: output format to the graph default=networkx (formats: networkx, adj_matrix, adj_list, adj_pandas)
        :param int language: language for text prepocessing, default=en (lang: en, es)
        :param bool apply_preprocessing: flag to exec text prepocessing, default=true
        :param bool parallel_exec: flag to exec tranformation in parallel, default=false
    """
    def __init__(self, 
                graph_type, 
                output_format='', 
                window_size=20, 
                parallel_exec=True, 
                language='en', 
                apply_preprocessing=True, 
                steps_preprocessing={},
                load_preprocessing=False
            ):
        """Constructor method
        """
        self.apply_prep = apply_preprocessing
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
        logger.debug('\t Getting windows')
        word_window_freq = defaultdict(int)  
        word_pair_count = defaultdict(int)
        len_doc_words_list = len(doc_words_list)
        len_windows = 0
        counter = 0

        for doc in doc_words_list:
            windows = []
            doc_words = doc['words']
            length = len(doc_words)

            counter += 1
            if counter == configs.NUM_PRINT_ITER:
                logger.debug("\t Iter %s out of %s", str(counter), str(len_doc_words_list))

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
        logger.debug('Get PMI measure')
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
        logger.debug('Get TF-IDF measure')
        vectorizer = TfidfVectorizer(vocabulary=vocab, norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False, lowercase=False, tokenizer=None)
        tfidf = vectorizer.fit_transform(corpus_docs_list)
        words_docs_tfids = []
        len_tfidf = tfidf.shape[0]
        counter = 0

        for ind, row in enumerate(tfidf):
            counter += 1
            if counter == configs.NUM_PRINT_ITER:
                logger.debug("\t Iter %s out of %s", str(counter), str(len_tfidf))
            for col_ind, value in zip(row.indices, row.data):
                edge = ('D-' + str(ind+1), vocab[col_ind], {'tfidf': round(value, 2)})
                words_docs_tfids.append(edge)
        return words_docs_tfids


    # normalize text
    def __text_normalize(self, text: dict) -> dict:
        text = self.prep.handle_blank_spaces(text)
        text = self.prep.handle_non_ascii(text)
        text = self.prep.handle_emoticons(text)
        text = self.prep.handle_html_tags(text)
        text = self.prep.handle_stop_words(text)
        text = self.prep.to_lowercase(text)
        text = self.prep.handle_blank_spaces(text)
        return text
    

    # get nodes an its attributes
    def __get_entities(self, corpus: list) -> list:  
        nodes = []
        for i in range(0, len(corpus)-1):
            node_doc =  ('D-' + str(i+1), {})
            nodes.append(node_doc)
            doc_words = self.prep.word_tokenize(corpus[i]['doc'])
            for word in doc_words:
                node_word = (str(word), {})
                nodes.append(node_word)
       

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
    def __build_graph(self, edges: list) -> networkx:
        # pending validations
        graph = super().set_graph_type(self.graph_type)
        #graph.add_nodes_from(nodes)
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
            logger.info("1. text preprocessing")
            corpus_docs_list = []
            doc_words_list = []
            len_corpus_docs = len(corpus_docs)
            vocab = set()
            counter = 0
            if not self.load_preprocessing: 
                logger.debug('\t Applying norm text')
                if self.apply_prep == True:
                    for i in range(len_corpus_docs):
                        corpus_docs[i]['doc'] = self.__text_normalize(corpus_docs[i]['doc'])
                        words = self.prep.word_tokenize(corpus_docs[i]['doc'])
                        doc_words_list.append({'doc': i, 'words': words})
                        corpus_docs_list.append(corpus_docs[i]['doc'])
                        vocab.update(set(words))
                        counter += 1
                        if counter == configs.NUM_PRINT_ITER:
                            logger.debug("\t Iter %s out of %s", str(counter), str(len_corpus_docs))
                    vocab = list(vocab)
                    self.utils.save_data(data=corpus_docs, path=configs.OUTPUT_DIR_HETERO_PATH, file_name='corpus_normalized', compress=1)
                    self.utils.save_data(data=vocab, path=configs.OUTPUT_DIR_HETERO_PATH, file_name='vocab', compress=1)
                else:
                    corpus_docs_list = corpus_docs
            else:
                logger.debug('\t Loading norm text')
                corpus_docs_list = self.utils.load_data(file_name='corpus_normalized', path=configs.OUTPUT_DIR_HETERO_PATH)
                corpus_docs_list = self.utils.load_data(file_name='vocab', path=configs.OUTPUT_DIR_HETERO_PATH)

            #2. build vocabulary
            #logger.info("2. build vocabulary")
            #doc_words_list, vocab = self.__build_vocab(corpus_docs)

            #3. get node/entities
            #logger.info("3. get node")
            #nodes = self.__get_entities(corpus_docs_list)

            #4. get edges/relations
            logger.info("2. get nodes/edges")
            edges = self.__get_relations(corpus_docs_list, doc_words_list, vocab)
            
            #5. build graph
            logger.info("3. build graph")
            graph = self.__build_graph(edges)
            output_dict['number_of_edges'] = graph.number_of_edges()
            output_dict['number_of_nodes'] = graph.number_of_nodes()
            #10. graph_transformation
            output_dict['graph'] = self.graph_trans.transform(self.output_format, graph)
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.debug('Error Detail: %s', str(traceback.format_exc()))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
        

    def transform(self, corpus_docs: list) -> list:
        logger.info("Init transformations: Text to Heterogeneous Graph")
        logger.info("Number of Text Documents: %s", len(corpus_docs))

        corpus_output_graph = [self.__transform_pipeline(corpus_docs)]
        self.utils.save_data(data=corpus_output_graph, path=configs.OUTPUT_DIR_HETERO_PATH, file_name='corpus_graph', compress=1)

        logger.info("Done transformations")
        return corpus_output_graph
    



