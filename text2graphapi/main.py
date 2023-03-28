import os
import sys
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import logging
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

'''
    Main file to run testing for Library
'''

# ****** SETTINGS:
# *** Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# *** Configs
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET = 'pan22' # posible values: pan14, pan15
TEST_API_FROM = 'LOCAL' #posible values: LOCAL, PYPI
PRINT_NUM_OUTPUT_GRAPHS = 5
INPUT_CORPUS_TEST = {
    'active': False,
    'corpus_text_docs': [
        #{'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'},
        {'id': 1, 'doc': "bible answers organization distribution"},
        {'id': 2, 'doc': "atheists agnostics organization"},
    ]
}


# TEST API PYPI
if TEST_API_FROM == 'PYPY':
    from text2graphapi.src.Cooccurrence  import Cooccurrence
# TEST API LOCAL
else:
    from src.Cooccurrence import Cooccurrence
    from src.Heterogeneous import Heterogeneous


def read_dataset(dataset, file):
    docs = []
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + dataset
    data = dataset_path + '/' + file
    for line in open(data, encoding='utf8'):
        docs.append(json.loads(line))
    return docs


def handle_spanish_fake_news_dataset(corpus_docs, num_rows=-1):
    id = 1
    new_corpus_docs = []
    for d in corpus_docs[:num_rows]:
        doc = {"id": id, "doc": d}
        new_corpus_docs.append(doc)
        id += 1
    return new_corpus_docs


def handle_20ng_dataset(corpus_docs, num_rows=-1):
    id = 1
    new_corpus_docs = []
    for d in corpus_docs[:num_rows]:
        doc = {"id": id, "doc": d}
        new_corpus_docs.append(doc)
        id += 1
    return new_corpus_docs


def handle_PAN_dataset(corpus_docs, num_rows=-1):
    new_corpus_docs = []
    docs_id = []
    for line in corpus_docs[:num_rows]:
        doc_text_1 = line['pair'][0]
        doc_text_2 = line['pair'][1]
        doc_id_1 = line['id'] + '_1'
        doc_id_2 = line['id'] + '_2'
        if (len(doc_text_1) == 0 or len(doc_text_2) == 0) or (line['id'] in docs_id):
            continue
        docs = [
            {"id": doc_id_1, "doc": doc_text_1},
            {"id": doc_id_2, "doc": doc_text_2}
        ]
        new_corpus_docs.extend(docs)
        docs_id.append(line['id'])
    return new_corpus_docs


def text_to_cooccur_graph(corpus_docs):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'DiGraph', 
            apply_preprocessing = True, 
            steps_preprocessing = {},
            parallel_exec = False,
            window_size = 2, 
            language = 'en', #es, en
            output_format = 'networkx'
        )
    # apply co_occur trnaformation
    corpus_cooccur_graphs = co_occur.transform(corpus_docs)
    return corpus_cooccur_graphs


def text_to_hetero_graph(corpus_docs):
    # create co_occur object
    hetero_graph = Heterogeneous(
        window_size = 20, 
        graph_type = 'Graph',
        parallel_exec = False,
        apply_preprocessing = True, 
        load_preprocessing = True, 
        steps_preprocessing = {},
        language = 'es', #es, en,
        output_format = 'networkx',
    )
    # apply Heterogeneous transformation
    corpus_hetero_graph = hetero_graph.transform(corpus_docs)
    return corpus_hetero_graph


def main():
    # read dataset

    '''DATASET = 'tass_emotion_detection'
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + DATASET
    dataset = dataset_path + '/emotion.csv'
    dataset_df = pd.read_csv(dataset, encoding= 'unicode_escape')
    dataset_list = dataset_df.to_dict('records')
    corpus_text_docs = []
    id = 1
    for d in dataset_list[:]:
        doc = {"id": id, "doc": d['texts']}
        corpus_text_docs.append(doc)
        id += 1'''


    '''DATASET = 'spanish_fake_news'
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + DATASET
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
        corpus_text_docs.append(doc)'''
    

    '''DATASET = '20_newsgroups'
    newsgroups_dataset = fetch_20newsgroups() #subset='train'
    corpus_text_docs = handle_20ng_dataset(newsgroups_dataset.data, num_rows=-1)   
    print(len(corpus_text_docs), corpus_text_docs[0])''' 


    if INPUT_CORPUS_TEST['active'] == True:
        logger.info("*** Reading dataset: INPUT_CORPUS_TEST")
        corpus_text_docs = INPUT_CORPUS_TEST['corpus_text_docs']
    else:
        logger.info("*** Reading dataset: %s", DATASET)
        train = read_dataset(DATASET, file='pan22-authorship-verification-training.jsonl')
        train_truth = read_dataset(DATASET, file='pan22-authorship-verification-training-truth.jsonl')
        #corpus.extend(read_dataset(DATASET, file='test.jsonl'))
        corpus_text_docs = handle_PAN_dataset(train, num_rows=-1)

    #print(len(corpus_text_docs), corpus_text_docs[0])
    
    # apply tranformations
    logger.info("*** Call API tranformation from: %s", TEST_API_FROM)
    start_time = time.time() # time init
    # expected input  ex: [{"id": 1, "doc": "text_data"}, ...]
    #corpus_graph_docs = text_to_cooccur_graph(corpus_text_docs) 
    corpus_graph_docs = text_to_hetero_graph(corpus_text_docs) 
    # expected output ex: [{"id": 1, "doc_graph": "adj_matrix", 'number_of_edges': 123, 'number_of_nodes': 321 'status': 'success'}, ...]
    end_time = (time.time() - start_time)
   
    # metrics
    logger.info("*** Results - Metrics: ")
    logger.info("Dataset: %s ", DATASET)
    logger.info('Num_Text_Docs: %i', len(corpus_text_docs))
    logger.info('Num_Graph_Docs: %i', len(corpus_graph_docs))
    logger.info("TOTAL TIME:  %s seconds" % end_time)

    # show corpus_graph_docs
    logger.info("*** Show Graphs Outputs: ")
    for graph in corpus_graph_docs[:PRINT_NUM_OUTPUT_GRAPHS]:
        print('\t', graph)
        #print(graph['graph'].nodes)
        #print(graph['graph'].edges)

if __name__ == '__main__':
    main()
