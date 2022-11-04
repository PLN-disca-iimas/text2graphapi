from src.models.Cooccurrence import Cooccurrence
import os
import json
import time


'''
    Main file to run testing for Library
'''

#corpus_input_texts = ["I go to school every day by bus.", 
#                      "i go to theatre every night by bus"]


root_path = os.path.dirname(os.path.dirname(__file__))
dataset = 'pan14'
dataset_path = root_path + '/text2graph-api/datasets/' + dataset
train_data = dataset_path + '/train.jsonl'
test_data = dataset_path + '/test.jsonl'

docs = []
for line in open(train_data, encoding='utf8'):
    docs.append(json.loads(line))
for line in open(test_data, encoding='utf8'):
    docs.append(json.loads(line))


num_problems = len(docs)
corpus_docs = []
for line in docs[:10]:
    doc = {
        "id": line['id'] + '_1', "doc": line['pair'][0],
        "id": line['id'] + '_2', "doc": line['pair'][1]
    }
    corpus_docs.append(doc)

occur = Cooccurrence(
        graph_type='DiGraph', 
        apply_prep=True, 
        window_size=1, 
       #output_format='adj_matrix'
    )

start_time = time.time() # time init
# expected input format: {"id": "doc_id", "doc": "text_data" ...}
corpus_output_texts = occur.transform(corpus_docs)
end_time = (time.time() - start_time)
print('--------------------------')
print('Dataset: ', dataset)
print('Num_Problems: ', num_problems)
print('Num_Docs: ', len(corpus_docs))
print("TOTAL TIME:  %s seconds" % end_time)

#print('corpus_texts: ', corpus_docs)
#print('corpus_graphs:')
for g in corpus_output_texts:
    print(g)


'''
corpus_output_texts = [
    {"doc_id": 1, "doc_text": ""}, 
    {"doc_id": 2, "doc_text": ""}, 
]
'''


'''
corpus_output_texts = [
    {"doc_id": 1, "doc_graph": "adj_matrix", 'status': 'success'}, 
    {"doc_id": 2, "doc_graph": "adj_matrix", 'status': 'success'}, 
]
'''