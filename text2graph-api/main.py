from src.models.Cooccurrence import Cooccurrence

'''
    Main file to run testing for Library
'''

corpus_input_texts = ["I go to school every day by bus.", 
                      "i go to theatre every night by bus"]

occur = Cooccurrence(
        graph_type='DiGraph', 
        apply_prep=False, 
        window_size=1, 
        output_format='adj_matrix'
    )
corpus_output_texts = occur.transform(corpus_input_texts)

print('corpus_texts: ', corpus_input_texts)
print('corpus_graphs', corpus_output_texts)


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