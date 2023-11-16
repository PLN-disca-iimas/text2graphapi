import os
import sys
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import logging
import argparse
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import glob
from functools import reduce

from src import configs # DEV, PROD
 
from itertools import chain
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')



''' 
    Main file to run testing for Library
'''

# ****** SETTINGS:
# *** Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# *** Configs
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PRINT_NUM_OUTPUT_GRAPHS = 5

logger.debug('Import libraries/modules from: %s', configs.ENV_EXECUTION)
# TEST API PROD
if configs.ENV_EXECUTION == 'PROD':
    from text2graphapi.src.Cooccurrence  import Cooccurrence
    from text2graphapi.src.Heterogeneous  import Heterogeneous
    from text2graphapi.src.IntegratedSyntacticGraph  import ISG
# TEST API DEV/LOCAL
else:
    from src.Cooccurrence import Cooccurrence
    from src.Heterogeneous import Heterogeneous
    from src.IntegratedSyntacticGraph import ISG


def read_dataset(dataset, file):
    docs = []
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + dataset
    data = dataset_path + '/' + file
    for line in open(data, encoding='utf8'):
        docs.append(json.loads(line))
    return docs


def read_custom_dataset():
    dataset_name = 'custom'
    logger.info("*** Using dataset: %s", dataset_name)
    corpus_text_docs = [
        {'id': 1, 'doc': 'I wonder if it has changed them back into real little children again'},
        {'id': 2, 'doc': 'My, my, I was forgetting all about the children and the mysterious fern seed'},
        {'id': 3, 'doc': 'Yes, here they come'}
        # {'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'},
        # {'id': 2, 'doc': "The 5 biggest countries by population in 2017 are China, India, USA, Indonesia, and Brazil."},
        # {'id': 3, 'doc': "Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls."},
    ]
    return corpus_text_docs


def read_spanish_fake_news_dataset():
    dataset_name = 'spanish_fake_news'
    logger.info("*** Using dataset: %s", dataset_name)
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + dataset_name
    dataset_train = dataset_path + '/train.csv'
    dataset_test = dataset_path + '/test.csv'
    train_df = pd.read_csv(dataset_train)
    test_df = pd.read_csv(dataset_test)
    train_list = train_df.to_dict('records')
    test_list = test_df.to_dict('records')
    corpus_docs = []
    corpus_text_docs = []
    corpus_docs.extend(train_list)
    corpus_docs.extend(test_list)
    for d in corpus_docs[:]:
        doc = {"id": d['id'], "doc": d['text']}
        corpus_text_docs.append(doc)
    return corpus_text_docs


def read_20_newsgroups_dataset():
    dataset_name = '20_newsgroups'
    logger.info("*** Using dataset: %s", dataset_name)
    newsgroups_dataset = fetch_20newsgroups() #subset='train', fetch from sci-kit learn
    id = 1
    corpus_text_docs = []
    for d in newsgroups_dataset.data[:]:
        doc = {"id": id, "doc": d}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs


def read_tass_emotion_detection_dataset():
    dataset_name = 'tass_emotion_detection'
    logger.info("*** Using dataset: %s", dataset_name)
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + dataset_name
    dataset = dataset_path + '/emotion.csv'
    dataset_df = pd.read_csv(dataset, encoding= 'unicode_escape')
    dataset_list = dataset_df.to_dict('records')
    corpus_text_docs = []
    id = 1
    for d in dataset_list[:]:
        doc = {"id": id, "doc": d['texts']}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs


def read_pan_dataset(dataset_name, file_name):
    logger.info("*** Using dataset: %s", dataset_name)
    dataset_dir = ROOT_DIR + '/text2graphapi/datasets/' + dataset_name
    files = glob.glob(f"{dataset_dir}/*.jsonl")
    df_files = [pd.read_json(path_or_buf=f, lines=True) for f in files]
    df_reduced = reduce(lambda df1,df2: pd.merge(df1,df2,how='left',on='id'), df_files)
    print(df_reduced.info())
    return handle_PAN_dataset(df_reduced)


def handle_PAN_dataset(corpus_df, num_rows=-1):
    corpus_df.drop_duplicates(subset="pair", keep='first', inplace=True)
    pairs_list = pd.Series([x for _list in pd.Series(corpus_df['pair']) for x in _list])    
    pairs_list = pairs_list.value_counts().index.tolist()
    text_list = []
    for i, p_list in enumerate(pairs_list):
        text_list.append({"id": i, "doc": p_list})
    return text_list


def text_to_cooccur_graph(corpus_docs):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'Graph', 
            apply_prep = True, 
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": True,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_negations": True,
                "handle_contractions": True,
                "handle_stop_words": True,
                "to_lowercase": True
            },
            parallel_exec = False,
            window_size = 3, 
            language = 'en', #spanish (sp), english (en), french (fr)
            output_format = 'networkx'
        )
    # apply co_occur trnaformation
    corpus_cooccur_graphs = co_occur.transform(corpus_docs)
    return corpus_cooccur_graphs

def text_to_isg_graph(corpus_docs):
    # Create ISG object
    isg = ISG(
            graph_type = 'Graph', 
            apply_prep = True, 
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": True,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_negations": True,
                "handle_contractions": True,
                "handle_stop_words": True,
                "to_lowercase": True
            },
            parallel_exec = False,
            language = 'en', #spanish (sp), english (en), french (fr)
            output_format = 'networkx'
        )
    # apply ISG trnaformation
    corpus_isg_graph = isg.transform(corpus_docs)
    return corpus_isg_graph


def text_to_hetero_graph(corpus_docs):
    # create co_occur object
    hetero_graph = Heterogeneous(
        window_size = 20, 
        graph_type = 'Graph',
        parallel_exec = False,
        apply_prep = True, 
        load_preprocessing = False, 
        steps_preprocessing = {},
        language = 'en', #spanish (sp), english (en), french (fr)
        output_format = 'networkx',
    )
    # apply Heterogeneous transformation
    corpus_hetero_graph = hetero_graph.transform(corpus_docs)
    return corpus_hetero_graph


def main(dataset, graph_type, cut_dataset=-1):
    # read dataset selected
    corpus_text_docs = None
    if dataset == 'tass_emotion_detection':
        corpus_text_docs = read_tass_emotion_detection_dataset()
    elif dataset == 'spanish_fake_news':
        corpus_text_docs = read_spanish_fake_news_dataset()
    elif dataset == '20_newsgroups':
        corpus_text_docs = read_20_newsgroups_dataset()
    elif dataset == 'pan_14':
        corpus_text_docs = read_pan_dataset(dataset_name='pan14', file_name='train.jsonl')
    elif dataset == 'pan_15':
        corpus_text_docs = read_pan_dataset(dataset_name='pan15', file_name='train.jsonl')
    elif dataset == 'pan_20':
        corpus_text_docs = read_pan_dataset(dataset_name='pan20', file_name='train_small.jsonl')
    elif dataset == 'pan_22':
        corpus_text_docs = read_pan_dataset(dataset_name='pan22', file_name='pan22-authorship-verification-training.jsonl')
    elif dataset == 'pan_23':
        corpus_text_docs = read_pan_dataset(dataset_name='pan23', file_name='pairs.jsonl')
    else:
        corpus_text_docs = read_custom_dataset()
    
    # cut dataset
    corpus_text_docs = corpus_text_docs[:cut_dataset]
    
    
    # set graph_type selcted & apply tranformations
    logger.info("*** Call API tranformation from: %s", configs.ENV_EXECUTION)
    start_time = time.time() # time init
    # expected input  ex: [{"id": 1, "doc": "text_data"}, ...]
    corpus_graph_docs = None
    if graph_type == 'Cooccurrence':
        corpus_graph_docs = text_to_cooccur_graph(corpus_text_docs) 
    if graph_type == 'Heterogeneous':
        corpus_graph_docs = text_to_hetero_graph(corpus_text_docs) 
    # expected output ex: [{"id": 1, "doc_graph": "adj_matrix", 'number_of_edges': 123, 'number_of_nodes': 321 'status': 'success'}, ...]
    if graph_type == 'ISG':
        corpus_graph_docs = text_to_isg_graph(corpus_text_docs)
    end_time = (time.time() - start_time)
   
    # metrics
    logger.info("*** Results - Metrics: ")
    logger.info('Num_Text_Docs: %i', len(corpus_text_docs))
    logger.info('Num_Graph_Docs: %i', len(corpus_graph_docs))
    logger.info("TOTAL TIME:  %s seconds" % end_time)

    # show corpus_graph_docs
    logger.info("*** Show Graphs Outputs: ")
    for graph in corpus_graph_docs[:PRINT_NUM_OUTPUT_GRAPHS]:
        print('\t', graph)
        print(graph['graph'].nodes)
        print(graph['graph'].edges)


if __name__ == '__main__':
    # datasets options  : default, tass_emotion_detection, spanish_fake_news, 20_newsgroups, pan_14, pan_15, pan_20, pan_22
    # graph_type options: Cooccurrence, Heterogeneous, ISG

    main(dataset='default', graph_type='ISG', cut_dataset=10)
    