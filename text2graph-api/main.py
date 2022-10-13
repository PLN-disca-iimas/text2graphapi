from src.models import Cooccurrence

'''
    Main file to run testing for Library
'''

corpus_texts = ["I go to school every day by bus.", "i go to theatre every night by bus"]
occur = Cooccurrence(
        graph_type='DiGraph', 
        apply_prep=True, 
        window_size=1, 
        output_format='adj_matrix'
    )
corpus_graphs = occur.transform(corpus_texts)

print('corpus_texts: ', corpus_texts)
print('corpus_graphs', corpus_graphs)