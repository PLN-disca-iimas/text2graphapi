.. _Heterogeneous:

Heterogeneous Graph
===========

In this graph, words and documents are represented as nodes and relation between word to word and word to document as edges. 
As an attributes/weights, the word to word relation has the point-wise mutual information (PMI) measure, 
and word to document relation has the Term Frequency-Inverse Document Frequency (TFIDF) measure. 
As output we will have only one grpah representation for all the text documents in the courpus.

Heterogeneous module
------------------------

.. automodule:: src.Heterogeneous
   :members:
   :undoc-members:
   :show-inheritance:

Example
-----------------------
In the following code snippet we have a corpus of two document, and we apply a Heterogeneous transformation 
with params: graph type as Graph, window_size of 20, English language, networkx object as desired output format, etc::

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


After the execution of this code, we have one undirected representing 
the whole corpus graph with 8 nodes and 11 edges::

   [{
      'id': 1, 
      'doc_graph': <networkx.classes.graph.Graph at 0x7f2b44e6d9a0>, 
      'number_of_edges': 11, 
      'number_of_nodes': 8, 
      'status': 'success'
   }]
