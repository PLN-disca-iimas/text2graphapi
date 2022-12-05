from src.Cooccurrence import Cooccurrence
import os
import json
import time
from text2graph_api.src.Cooccurrence  import Cooccurrence


'''
    Main file to run testing for Library
'''

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def read_dataset(dataset, file):
    docs = []
    dataset_path = ROOT_DIR + '/text2graph-api/datasets/' + dataset
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
            window_size = 1, 
            language = 'en', #es, en
            output_format = 'adj_matrix'
        )
    # apply co_occur trnaformation
    corpus_cooccur_graphs = co_occur.transform(corpus_docs)
    return corpus_cooccur_graphs


def main():
    # read dataset
    dataset = 'pan14'
    corpus = read_dataset(dataset, file='train.jsonl')
    corpus.extend(read_dataset(dataset, file='test.jsonl'))

    # handle PAN datsets: format normalization
    corpus_text_docs = handle_PAN_dataset(corpus, num_rows=10)
    
    start_time = time.time() # time init
    # expected input  ex: [{"id": 1, "doc": "text_data"}, ...]
    # expected output ex: [{"id": 1, "doc_graph": "adj_matrix", 'number_of_edges': 123, 'number_of_nodes': 321 'status': 'success'}, ...]
    corpus_graph_docs = text_to_cooccur_graph(corpus_text_docs) 
    end_time = (time.time() - start_time)
   
    # metrics
    print('--------------------------')
    print('Dataset: ', dataset)
    print('Num_Problems: ', len(corpus))
    print('Num_Text_Docs: ', len(corpus_text_docs))
    print('Num_Graph_Docs: ', len(corpus_graph_docs))
    print("TOTAL TIME:  %s seconds" % end_time)

    # show 5 first corpus_graph_docs
    for graph in corpus_graph_docs[:5]:
        print(graph)


if __name__ == '__main__':
    main()
