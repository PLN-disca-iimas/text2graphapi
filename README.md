
# text2graph Library

**text2graphapi** is a python library for text-to-graph tranformations. To use this library it is necessary to install the modules and dependencies in user’s application. Also, the corpus of text documents to be transformed into graphs has to be loaded and read.

 The  text to graph transformation pipeline consists of three main modules:
* **Text Preprocessing and Normalization**. This module aims to perform all the cleaning and pre-processing part of the text. Handling blank spaces, emoticons, HTML tags stop words, etc
* **Graph Model**. This module aims to define the entities/nodes and their relationships/edges according to the problem specification. 
* **Graph Transformation**. This module aims to apply vector transformations to the graph as final output, such as adjacency matrix, dense matrix, etc.

The following diagram depicts the pipeline overview for the text to graph tranformation described above:

![texto to graph pipeline](https://www.linkpicture.com/q/texto-to-graph.pipeline.png#center)

## **_Installation_ from PYPI**
Inside your project, from your CLI type the following command in order to install the latest version of the library:
```Python
pip install text2graphapi
```

## **_Types of graph representation available:_**
Currently, this library support two types of graph representation: *Word Co-Ocurrence Graph* and  *Heterogeneous Graph*. For both representation, the expected input is the same, and has to be the following structure:
```Python
# The input has to be a list of dictionaries, where ecah dict conatins an 'id' and 'doc' text data
# For example:
input_text_docs = [
	{"id": 1, "doc": "text for document 1"},
    {"id": 2, "doc": "text for document 2"}
]
```

I the netxt sections we decribe each of this graph representations and provide some implementation examples:
 - **Word Co-Ocurrence Graph:**
   In this graph, words are represented as a node and the co-occurence of two words within the document text is represented as an edge between the words/nodes. As an attributes/weights, nodes has *Part Of Speech* tag and egdes has the *number of co-occurrence*  between words in the text document. As output we will have one grpah representation for each text document  in the courpus.
   For example, in the following code snippet we have a corpus of one document, and we apply a word-occurence transformation with params: graph type as Digraph, window_size of 1, English language, adjacency matrix as desired output format, etc
   
```Python
from text2graphapi.src.Cooccurrence import Cooccurrence

corpus_docs = [{'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'}]

to_cooccurrence = Cooccurrence(
                graph_type = 'DiGraph', 
                apply_prep = True, 
                parallel_exec = False,
                window_size = 1, 
                language = 'en',
                output_format = 'adj_matrix')
                
output_text_graphs = to_cooccurrence.transform(corpus_docs)
```
After the execution of this code, we have one directed graph with 8 nodes and 15 edges:
```Python
[{
	'doc_id': 1, 
	'graph': <8x8 sparse array of type '<class 'numpy.int64'>' Sparse Row format>, 
	'number_of_edges': 15, 
	'number_of_nodes': 8, 
	'status': 'success'
}]
```

- **Heterogeneous Graph:**
In this graph, words and documents are represented as nodes and relation between word to word and word to document as edges. As an attributes/weights, the word to word relation has the point-wise mutual information (PMI) measure, and word to document relation has the Term Frequency-Inverse Document Frequency (TFIDF) measure. As output we will have only one grpah representation for all the text documents in the courpus.
For example, in the following code snippet we have a corpus of two document, and we apply a Heterogeneous transformation with params: graph type as Graph, window_size of 20, English language, networkx object as desired output format, etc
```Python
from text2graphapi.src.Heterogeneous import Heterogeneous

corpus_docs= [
	{'id': 1, 'doc': "bible answers organization distribution"},
	{'id': 2, 'doc': "atheists agnostics organization"},
]

hetero_graph = Heterogeneous(
				graph_type = 'Graph',
		        window_size = 20, 
		        parallel_exec = False,
		        apply_preprocessing = True, 
		        language = 'es',
		        output_format = 'networkx')

output_text_graphs = hetero_graph.transform(corpus_docs)
```

After the execution of this code, we have one undirected representing the whole corpus graph with 8 nodes and 11 edges:
```Python
[{
	'id': 1, 
	'doc_graph': <networkx.classes.graph.Graph at 0x7f2b44e6d9a0>, 
	'number_of_edges': 11, 
	'number_of_nodes': 8, 
	'status': 'success'
}]
```
## **Acknowledgments**
This work has been carried out with the support of DGAPA UNAM-PAPIIT projects TA101722, IT100822, and CONAHCYT CF-2023-G-64.  The authors also thank CONAHCYT for the computing resources provided through the Deep Learning Platform for Language Technologies of the INAOE Supercomputing Laboratory.
