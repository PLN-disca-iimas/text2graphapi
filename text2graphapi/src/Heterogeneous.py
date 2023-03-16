import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from math import log
from joblib import Parallel, delayed

from text2graphapi.src.Utils import Utils
from text2graphapi.src.Preprocessing import Preprocessing
from text2graphapi.src.GraphTransformation import GraphTransformation
from text2graphapi.src import Graph

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
    def __init__(self, graph_type, output_format='', apply_preprocessing=True, window_size=20, parallel_exec=True, language='en', steps_preprocessing={}):
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


    # build vocab
    # get word set and word frequency
    def __build_vocab(self, corpus):
        word_set = set()
        word_freq = {}
        for i in range(len(corpus)):
            doc_words = corpus[i]['doc']
            words = self.prep.word_tokenize(doc_words)
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        return list(word_set), word_freq
    

    # get word_id_map
    # assign intetger ID to each word in vocabulary -> {w1: 1, w2: 2, ...}
    def __get_word_id_map(self, vocab):
        word_id_map = {}
        for i in range(len(vocab)):
            word_id_map[vocab[i]] = i
        return word_id_map


    # get word_doc_freq
    # frequency for each word in docs -> {w1: 20, w2: 3', ...} 
    def __get_word_doc_freq(self, word_doc_list):
        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)
        return word_doc_freq


    # get word_doc_list
    # list docs where appear each word -> {w: [d1,d2,...], w2: [d3,d4,...], ...}
    def __get_word_doc_list(self, corpus):
        word_doc_list = {}
        for i in range(len(corpus)):
            doc_words = corpus[i]['doc']
            words = self.prep.word_tokenize(doc_words)
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)
        return word_doc_list


    # get windows
    # words windows based on window_size param -> [[w1,w2,...], [w3,w4,...], ...]
    def __get_windows(self, corpus, window_size):
        windows = []
        for i in range(len(corpus)):
            doc_words = corpus[i]['doc']
            words = self.prep.word_tokenize(doc_words)
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
        return windows

    # get windows word frequency
    # word frequency in windows -> {w1: 4, w2: 5,  ...}
    def __get_word_window_freq(self, windows):
        word_window_freq = {}
        for window in windows[:]:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])
        return word_window_freq


    # get word pair count in windows
    # word frequency in windows -> {'w1,w2': 4, 'w5,w6': 5,  ...}
    def __get_word_pair_count(self, windows, word_id_map):
        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
        return word_pair_count


    # get pmi measure
    # pmi for pair of word,word -> {'w1,w2': pmi, 'w5,w6': pmi,  ...}
    def __get_pmi(self, word_pair_count, word_window_freq, vocab, window_size):
        word_to_word_pmi = []
        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / window_size) / (1.0 * word_freq_i * word_freq_j/(window_size * window_size)))
            if pmi <= 0:
                continue
            word_to_word_pmi.append({key: pmi})
        return word_to_word_pmi      

    # get doc word frequency
    # freq of pair doc,word -> {'doc1,word1': 3, 'doc1,word2': 3,  ...}
    def __get_doc_word_frequency(self, corpus, word_id_map):
        doc_word_freq = {}
        for doc_id in range(len(corpus)):
            doc_words = corpus[doc_id]['doc']
            words = self.prep.word_tokenize(doc_words)
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1
        return doc_word_freq


    # get tfidf meausure
    # tfid for relation doc,word -> {'doc1,word1': tfidf, 'doc1,word2': tfidf,  ...}
    def __get_tfidf(self, corpus, word_id_map, doc_word_freq, word_doc_freq, vocab):
        word_to_doc_tfid = []
        for i in range(len(corpus)):
            doc_words = corpus[i]['doc']
            words = self.prep.word_tokenize(doc_words)
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                idf = log(1.0 * len(corpus) /  word_doc_freq[vocab[j]])
                word_to_doc_tfid.append({key: freq * idf})
                doc_word_set.add(word)
        return word_to_doc_tfid


    # normalize text
    def __text_normalize(self, text: str) -> list:
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
        return nodes


   # get edges an its attributes
    def __get_relations(self, words_pmi, words_docs_tfids, vocab) -> list:  
        edges = []
        for w_w_pmi in words_pmi:
            for key, value in w_w_pmi.items():
                word_1, word_2 = key.split(',')
                word_1 = vocab[int(word_1)]
                word_2 = vocab[int(word_2)]
                edge = (word_1, word_2, {'pmi': round(value, 2)})  # (word_i, word_j, {'pmi': value})
                edges.append(edge) 
        for w_d_tfidf in words_docs_tfids:
            for key, value in w_d_tfidf.items():
                doc, word = key.split(',')
                word = vocab[int(word)]
                edge = ('D-' + str(doc), word, {'tfidf': round(value, 2)})  # (doc, word, {'tfidf': value})
                edges.append(edge) 
        return edges


    # build nx-graph based of nodes and edges
    def __build_graph(self, nodes: list, edges: list) -> networkx:
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
            if self.apply_prep == True:
                for i in corpus_docs:
                    i['doc'] = self.__text_normalize(i['doc'])
            #2. build vocabulary
            vocab, word_freq = self.__build_vocab(corpus_docs)
            #3. get word_doc_list AND word_doc_freq AND word_id_map
            word_doc_list = self.__get_word_doc_list(corpus_docs)
            word_doc_freq = self.__get_word_doc_freq(word_doc_list)
            word_id_map = self.__get_word_id_map(vocab)
            #4. get windows AND word window frequency
            windows = self.__get_windows(corpus_docs, self.window_size)
            word_window_freq = self.__get_word_window_freq(windows)
            #5. get word pair count AND pmi measure
            word_pair_count = self.__get_word_pair_count(windows, word_id_map)
            words_pmi = self.__get_pmi(word_pair_count, word_window_freq, vocab, len(windows))
            #6. get doc word frequency AND tfidf measure
            doc_word_frequency = self.__get_doc_word_frequency(corpus_docs, word_id_map)
            words_docs_tfids = self.__get_tfidf(corpus_docs, word_id_map, doc_word_frequency, word_doc_freq, vocab)
            #7. get node/entities
            nodes = self.__get_entities(corpus_docs)
            #8. get edges/relations
            edges = self.__get_relations(words_pmi, words_docs_tfids, vocab)
            #9. build graph
            graph = self.__build_graph(nodes, edges)
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
        logger.info("Done transformations")
        return corpus_output_graph
    



