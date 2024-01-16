
  

# text2graph Library


**text2graphapi** is a python library for text-to-graph tranformations. To use this library it is necessary to install the modules and dependencies in user’s application. Also, the corpus of text documents to be transformed into graphs has to be loaded and read.  

The text to graph transformation pipeline consists of three main modules, ad depict in the following diagram:

![texto to graph pipeline](https://iili.io/JTwzuXj.png#center)

*  **Text Preprocessing and Normalization**. This component receives the input text (in a specific format/structure) and performs all the cleaning and normalization steps for the data. It applies different text cleaning methods such as removing stop words, handling contractions, handling ASCI characters, and so on. Moreover, it performs different NLP techniques such as POS tags, tokenization, lemmatization, etc (using third-party libraries such as Spacy, and NLTK).

*  **Graph Model**. This second component aims to define and construct the entities/nodes and their relationships/edges from the corpus texts to generate the specified graph representation. Currently, this library supports three text-to-graph transformations: Word-Coocurrence, Heterogeneous Graph, and Integrated Syntactic Graph (ISG). We will see each of them in detail in the following sections.

*  **Graph Transformation**. This final module receives the generated graph as an input (set of nodes and edges) and applies vector transformations to obtain the final graph representation as an output. This graph output is specified in the input parameters and supports different formats such as adjacency list, adjacency matrix, dense matrix, networkx object, etc.
  

## **_Installation_ from PYPI**

Inside your project, from your CLI type the following command in order to install the latest version of the library:

```Python

pip install text2graphapi

```
  

## **_Types of graph representation available:_**

Currently, this library support three types of graph representation: **Word Co-Ocurrence Graph**, **Heterogeneous Graph** and **Integrated Syntactic Graph**.   The Word Co-Occurrence transformations are classified as Document-level graphs due to there is one output graph per one input document (obtain one graph for each document in the corpus), and the Heterogeneous and ISG transformations are classified as Corpus-level graphs due to there is one output graph to represent the whole corpus (obtain one graph for all document in the corpus). 

The following code snippet shows a basic example using the text2graphapi library fot these three repsentation.

```Python

# The input has to be a list of dictionaries, where ecah dict conatins an 'id' and 'doc' text data

from text2graphapi.src.Cooccurrence import Cooccurrence
from text2graphapi.src.Heterogeneous import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph import ISG

corpus_docs = [
    {'id': 1, 'doc': "The sun was shining, making the river look bright and happy."},
    {'id': 2, 'doc': "Even with the rain, the sun came out a bit, making the wet river shine."}]

to_word_coocc_graph = Cooccurrence(graph_type = 'DiGraph', 
        language = 'en', apply_preprocessing = True, 
        window_size = 3, output_format = 'adj_matrix')

to_hetero_graph = Heterogeneous(graph_type = 'Graph', 
        window_size = 20, apply_preprocessing = True, 
        language = 'en', output_format = 'networkx')

to_isg_graph = ISG(graph_type = 'DiGraph',  language = 'en', 
        apply_preprocessing = True, output_format = 'networkx')

to_hetero_graph.transform(corpus_docs)
to_word_coocc_graph.transform(corpus_docs)
to_isg_graph.transform(corpus_docs)

```

In the next section, we will see some illustrative examples generated for this code. We will show each of the graph representations and explain in detail how they are built.

-  **Word Co-Ocurrence Graph:**
In this graph, words are represented as a node, and the co-occurrence of two words within the document text is represented as an edge between the words/nodes. As attributes/weights, nodes have the POS tag, and edges have the number of co-occurrences between words in the text document. As output, we will have one graph representation for each text document in the corpus.

![Cooc Graph](https://iili.io/JTw0xkv.png#center)


-  **Heterogeneous Graph:**
 In this graph, words and documents are represented as nodes, and the relation between word to word and word to document as edges. As attributes/weights, the word-to-word relation has the point-wise mutual information (PMI) measure, and the word-to-document relation has the Term Frequency-Inverse Document Frequency (TFIDF) measure. As output, we will have only one graph representation for all the text documents in the corpus
 
![Hetero Graph](https://iili.io/JTwaj4a.png#center)
  
.  
-  **Integrated Syntactic Graph:**

This representation, integrates multiple linguistic levels in a single data structure. These levels are: the Lexical level (lexical items such as words), Morphological level(deals with the identification, analysis, and description of the structure of the given language’s morphemes such as POS, roots, stem, etc), Syntactic level (deals with the sentence structure such as the dependency trees), and Semantic level (deals with the meaning of the sentences, this can include antonymy, synonymy, etc).

![ISG Graph](https://iili.io/JTw5N24.png#center)


  