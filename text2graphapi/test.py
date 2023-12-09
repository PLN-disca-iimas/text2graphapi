import os
import sys
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import logging
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import argparse
import glob
from functools import reduce
import pickle 
import datetime
import re

from src import configs # DEV, PROD

'''
    Main file to run testing for Library
'''

# ****** SETTINGS:
# *** Logging configs
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# *** Configs
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TODAY_DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
PRINT_NUM_OUTPUT_GRAPHS = 5
LANGUAGE = 'en' #es, en, fr

# TEST API PYPI
if configs.ENV_EXECUTION == 'PROD':
    from text2graphapi.src.Cooccurrence  import Cooccurrence
    from text2graphapi.src.Heterogeneous  import Heterogeneous
    from text2graphapi.src.IntegratedSyntacticGraph  import ISG
# TEST API LOCAL
else:
    from src.Cooccurrence import Cooccurrence
    from src.Heterogeneous import Heterogeneous
    from src.IntegratedSyntacticGraph import ISG


def set_logger(dataset, graph_type):
    log_filename = f'{dataset}_{graph_type}_{TODAY_DATE}.log' 
    file_handler = logging.FileHandler(known_args['output_path'] + log_filename, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)    
    logger = logging.getLogger()  # root logger
    logger.addHandler(file_handler)
    

def save_data(path, filename, data):
    with open(path + filename + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def read_custom_dataset():
    corpus_text_docs = [
        {'id': 1, 'doc': 'I wonder if it has changed them back into real little children again'},
        {'id': 2, 'doc': 'My, my, I was forgetting all about the children and the mysterious fern seed'},
        {'id': 3, 'doc': 'Yes, here they come back!'}
        # {'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'},
        # {'id': 2, 'doc': "The 5 biggest countries by population in 2017 are China, India, USA, Indonesia, and Brazil."},
        # {'id': 3, 'doc': "Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls."},
    ]
    return corpus_text_docs


def read_pan_dataset(path, dataset_name='pan23', files_pattern = '/*.jsonl'):
    corpus_text_docs = []
    dataset_path = path + dataset_name + files_pattern
    files = glob.glob(dataset_path)
    df_files = [pd.read_json(path_or_buf=f, lines=True) for f in files]
    df_courpus_reduced = reduce(lambda df1,df2: pd.merge(df1,df2,how='left',on='id'), df_files)
    #df_courpus_reduced.drop_duplicates(subset="pair", keep='first', inplace=True)    
    pairs_list = pd.Series([x for _list in pd.Series(df_courpus_reduced['pair']) for x in _list])    
    pairs_list = pairs_list.value_counts().index.tolist()
    for i, p_list in enumerate(pairs_list):
        corpus_text_docs.append({"id": i, "doc": p_list})
    return corpus_text_docs


def read_spanish_fake_news_dataset(path, dataset_name='spanish_fake_news'):
    dataset_path = path + dataset_name
    dataset_train = dataset_path + '/train.csv'
    dataset_test = dataset_path + '/test.csv'
    train_df = pd.read_csv(dataset_train)
    test_df = pd.read_csv(dataset_test)
    train_list = train_df.to_dict('records')
    test_list = test_df.to_dict('records')
    corpus_docs = train_list + test_list
    corpus_text_docs = []
    for d in corpus_docs[:]:
        doc = {"id": d['id'], "doc": d['text']}
        corpus_text_docs.append(doc)
    return corpus_text_docs


def read_20_newsgroups_dataset(path, dataset_name='20ng'):
    newsgroups_dataset = fetch_20newsgroups() #subset='train', fetch from sci-kit learn
    id = 1
    corpus_text_docs = []
    for d in newsgroups_dataset.data[:]:
        doc = {"id": id, "doc": d}
        corpus_text_docs.append(doc)
        id += 1
    return corpus_text_docs

def read_french_tgb_dataset(path, dataset_name='french_tgb', files_pattern = '/*.json'):
    corpus_text_docs = []
    dataset_path = path + dataset_name + files_pattern
    dataset_json_docs = glob.glob(dataset_path)
    for json_file in dataset_json_docs[:100]:
        f = open(json_file)
        file_data = json.load(f)
        corpus_text_docs.append({
            "id": file_data['doc_id'], 
            "doc": re.sub(r'\s+', ' ', file_data['text'])
        })
    return corpus_text_docs


def read_tass_emotion_detection_dataset():
    ''' PENDING   
    DATASET = 'tass_emotion_detection'
    dataset_path = '/002/usuarios/andric.valdez/andric/datasets/' + DATASET
    dataset = dataset_path + '/emotion.csv'
    dataset_df = pd.read_csv(dataset, encoding= 'unicode_escape')
    dataset_list = dataset_df.to_dict('records')
    corpus_text_docs = []
    id = 1
    for d in dataset_list[:]:
        doc = {"id": id, "doc": d['texts']}
        corpus_text_docs.append(doc)
        id += 1
    '''

def cooccur_graph_instance(lang='en'):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'DiGraph', 
            window_size = 2, 
            apply_prep = True,
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": True,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_contractions": True,
                "handle_stop_words": True,
                "to_lowercase": True
            },
            parallel_exec = False,
            language = lang, #es, en, fr
            output_format = 'networkx'
        )
    return co_occur


def hetero_graph_instance(lang='en'):
    # create co_occur object
    hetero_graph = Heterogeneous(
        graph_type = 'DiGraph',
        window_size = 10, 
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        load_preprocessing = False, 
        language = lang, #sp, en, fr
        output_format = 'networkx',
    )
    return hetero_graph


def isg_graph_instance(lang='en'):
    # create isg object
    isg = ISG(
        graph_type = 'DiGraph',
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        language = lang, #spanish (sp), english (en), french (fr)
        output_format = 'networkx'
    )
    return isg


def get_dataset(path, dataset):
    # read dataset selected
    corpus_text_docs = None
    if dataset == 'tass_emotion_detection':
        return read_tass_emotion_detection_dataset()
    elif dataset == 'spanish_fake_news':
        return read_spanish_fake_news_dataset(path=path, dataset_name=dataset)
    elif dataset == 'french_tgb':
        return read_french_tgb_dataset(path, dataset_name='french_tgb', files_pattern = '/clean_json_data/*.json')
    elif dataset == '20ng':
        return read_20_newsgroups_dataset(path=path, dataset_name=dataset)
    elif dataset == 'pan14':
        return read_pan_dataset(path=path, dataset_name=dataset)
    elif dataset == 'pan15':
        return read_pan_dataset(path=path, dataset_name=dataset)
    elif dataset == 'pan20':
        return read_pan_dataset(path=path, dataset_name=dataset)
    elif dataset == 'pan22':
        return read_pan_dataset(path=path, dataset_name=dataset, files_pattern="/pan22-authorship-verification-training.jsonl")
    elif dataset == 'pan23':
        return read_pan_dataset(path=path, dataset_name=dataset)
    else:
        return read_custom_dataset()

    
def get_text_2_graph_instance(graph_type, lang='en'):
    if graph_type == 'cooccurrence':
        g = cooccur_graph_instance(lang)
        return cooccur_graph_instance(lang)
    elif graph_type == 'heterogeneous':
        return hetero_graph_instance(lang)
    elif graph_type == 'isg':
        return isg_graph_instance(lang)
    else:
        ...
    
    
def main(args):
    path = args['dataset_path']    
    dataset = args['dataset_name']
    graph_type = args['graph_type']
    output_path = args['output_path']
    dataset_language = args['dataset_language']
    save_logs = args['save_logs'] == 'true'
    save_graph = args['save_graph'] == 'true'
    cut_percentage_dataset = args['cut_percentage_dataset']
    
    # set logger
    if save_logs == True:
        set_logger(dataset, graph_type)
    
    logger.info("*** TEST: Input CLI params %s", str(args))
    
    # read/get dataset selected
    logger.info("*** TEST: Reading dataset from: %s", path + dataset)
    corpus_text_docs = get_dataset(path, dataset)
    
    # get instance from text2graph library
    logger.info("*** TEST: Using graph type %s", graph_type)
    graph_instance = get_text_2_graph_instance(graph_type, lang=dataset_language)
    logger.info("\t * Graph config: %s", str(vars(graph_instance)))
    # apply text to graph transformation
    logger.info("*** TEST: Call API tranformation from %s", configs.ENV_EXECUTION)
    start_time = time.time() # time init
    # expected:
        # input corpus_text_docs: [{"id": 1, "doc": "text_data"}, ...]
        # output graph_output: [{"id": 1, "doc_graph": "adj_matrix", 'number_of_edges': 123, 'number_of_nodes': 321 'status': 'success'}, ...]
        
    cut_dataset = len(corpus_text_docs) * (int(cut_percentage_dataset) / 100)
    graph_output = graph_instance.transform(corpus_text_docs[:int(cut_dataset)])
    end_time = (time.time() - start_time)
    
    # get and save metrics
    logger.info("*** TEST: Results - Metrics ")
    logger.info("\t * Dataset: %s ", dataset)
    logger.info('\t * Num__total_Text_Docs: %i', len(corpus_text_docs))
    logger.info('\t * Num_Text_Docs_Used: %i', cut_dataset)
    logger.info('\t * Num_Graph_Docs_Output: %i', len(graph_output))
    logger.info("\t * TOTAL TIME:  %s seconds" % end_time)

    # show corpus_graph_docs
    logger.info("*** TEST: Show Graphs Outputs ")
    for graph in graph_output[:PRINT_NUM_OUTPUT_GRAPHS]:
        logger.info('\t %s', str(graph))
        #print(graph['graph'].nodes(data=True))
        #print(graph['graph'].edges(data=True))
    
    if save_graph == True:
        graph_filename = f'{dataset}_{graph_type}_{TODAY_DATE}' 
        save_data(output_path, graph_filename, graph_output)

 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", help="dataset path to use in graph tranformations", default='/002/usuarios/andric.valdez/andric/datasets/', type=str)
    parser.add_argument("-op", "--output_path", help="uutput path to save results", default='/002/usuarios/andric.valdez/andric/projects/text2graph-API/text2graphapi/outputs/', type=str)
    parser.add_argument("-dn", "--dataset_name", help="dataset name to use in graph tranformations", default='custom', type=str)
    parser.add_argument("-dl", "--dataset_language", help="language of the dataset: English, Spanish or French", default='en', type=str)
    parser.add_argument("-cpd", "--cut_percentage_dataset", help="percentage of instances to use in the dataset: from 0 to 100 % (default)", default='100', type=str)
    parser.add_argument("-gt", "--graph_type", help="graph transformation type to use", default='cooccurrence', type=str)
    parser.add_argument("-sl", "--save_logs", help="print and save logs", default='false', type=str)
    parser.add_argument("-sg", "--save_graph", help="save graph generated as pkl", default='false', type=str)
    known_args, unknown_args = parser.parse_known_args()
    known_args = vars(known_args)
    main(known_args)
    
    # *** Useful commands
    # nohup python test.py -dn=pan23 -gt=isg -cpd=10 &
    # python test.py -dn=20ng -gt=isg -cpd=10 -sl=false -sg=false
    # python test.py -dn=20ng -gt=isg -cpd=10 -dp=/c/Users/anvaldez/Documents/Docto/Projects/text2graph-API/text2graphapi/datasets/ -op=/c/Users/anvaldez/Documents/Docto/Projects/text2graph-API/text2graphapi/outputs/
    # ps -ef | grep python | grep andric
    # conda activate ../../../../anaconda3/envs/text2graphapi/
    
    # *** Input options params:
    # datasets options  : custom, tass_emotion_detection, spanish_fake_news, 20ng, pan_14, pan15, pan20, pan22, pan23, 
    # graph_type options: isg, cooccurrence, heterogeneous