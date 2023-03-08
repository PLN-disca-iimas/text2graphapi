import os
import sys
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import logging

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
DATASET = 'pan14' # posible values: pan14, pan15
TEST_API_FROM = 'LOCAL' #posible values: LOCAL, PYPI
PRINT_NUM_OUTPUT_GRAPHS = 5
INPUT_CORPUS_TEST = {
    'active': True,
    'corpus_text_docs': [
        {'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'},
    ]
}

# TEST API LOCAL
if TEST_API_FROM == 'PYPY':
    from text2graphapi.src.Cooccurrence  import Cooccurrence
# TEST API PYPI
else:
    from src.Cooccurrence import Cooccurrence 


def read_dataset(dataset, file):
    docs = []
    dataset_path = ROOT_DIR + '/text2graphapi/datasets/' + dataset
    data = dataset_path + '/' + file
    for line in open(data, encoding='utf8'):
        docs.append(json.loads(line))
    return docs


def handle_PAN_dataset(corpus_docs, num_rows=-1):
    new_corpus_docs = []
    for line in corpus_docs[:num_rows]:
        docs = [
            {"id": line['id'] + '_1', "doc": line['pair'][0]},
            {"id": line['id'] + '_2', "doc": line['pair'][1]}
        ]
        new_corpus_docs.extend(docs)
    return new_corpus_docs


def text_to_cooccur_graph(corpus_docs):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'DiGraph', 
            apply_prep = True, 
            parallel_exec = False,
            window_size = 3, 
            language = 'en', #es, en
            output_format = 'adj_matrix'
        )
    # apply co_occur trnaformation
    corpus_cooccur_graphs = co_occur.transform(corpus_docs)
    return corpus_cooccur_graphs


def main():
    # read dataset

    if INPUT_CORPUS_TEST['active'] == True:
        DATASET = 'INPUT_CORPUS_TEST'
        logger.info("*** Reading dataset: %s", DATASET)
        corpus_text_docs = INPUT_CORPUS_TEST['corpus_text_docs']
    else:
        logger.info("*** Reading dataset: %s", DATASET)
        corpus = read_dataset(DATASET, file='train.jsonl')
        corpus.extend(read_dataset(DATASET, file='test.jsonl'))
        corpus_text_docs = handle_PAN_dataset(corpus, num_rows=10)
    
    # apply tranformations
    logger.info("*** Call API tranformation from: %s", TEST_API_FROM)
    start_time = time.time() # time init
    # expected input  ex: [{"id": 1, "doc": "text_data"}, ...]
    corpus_graph_docs = text_to_cooccur_graph(corpus_text_docs) 
    # expected output ex: [{"id": 1, "doc_graph": "adj_matrix", 'number_of_edges': 123, 'number_of_nodes': 321 'status': 'success'}, ...]
    end_time = (time.time() - start_time)
   
    # metrics
    logger.info("*** Results - Metrics: ")
    logger.info("Dataset: %s ", DATASET)
    logger.info('Num_Text_Docs: %i', len(corpus_text_docs))
    logger.info('Num_Graph_Docs: %i', len(corpus_graph_docs))
    logger.info("TOTAL TIME:  %s seconds" % end_time)

    # show corpus_graph_docs
    for g in corpus_graph_docs[:PRINT_NUM_OUTPUT_GRAPHS]:
        print(g)

if __name__ == '__main__':
    main()
