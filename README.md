# text2graph-API
Use this library for text-to-graph tranformations. To use the API it is necessary to install its modules and dependencies  in the user’s application. Also, the corpus of text documents to be transformed into graphs has to be loaded and read.

 **text2graphapi** is a text to graph transformation pipeline that consists of four main modules::
* **Text Preprocessing and Normalization**. This module aims to
perform all the cleaning and pre-processing part of the text. Apply NLP methods such as POS-Tag, Lemm, Stem, etc.
* **Graph Model**. This module aims to define the entities/nodes and
their relationships/edges according to the problem specification.
* **Graph Extraction**. This module aims to build the graph according
to the selected model. We use third-party libraries such as NetworkX.
* **Graph Transformation and Analysis**. This module aims to apply
vector transformations to the graph as final output, such as adjacency
matrix, dense matrix, etc.


**_Where to get it_**
```Python
# from PYPI
pip install text2graphapi
```

**_Example input data_**
```Python
# Has to be a list of dict, where ecah dict conatins an 'id' and 'doc' text data
input_text_docs = [{"id": 1, "doc": "text_data_1"},
                   {"id": 2, "doc": "text_data_2"}]
```

**_How to use it_**
```Python
from text2graphapi.src.Cooccurrence import Cooccurrence

to_cooccurrence = Cooccurrence(
                graph_type = 'DiGraph', 
                apply_prep = True, 
                parallel_exec = False,
                window_size = 1, 
                language = 'en',
                output_format = 'adj_matrix')
                
output_text_graphs = to_cooccurrence.transform(corpus_docs)
```