.. _Cooccurrence:


Word Cooccurrence Graph
===========

In this graph, words are represented as a node and the co-occurence of two words 
within the document text is represented as an edge between the words/nodes. 
As an attributes/weights, nodes has Part Of Speech tag and egdes has the number of co-occurrence between words in the text document. 
As output we will have one grpah representation for each text document in the courpus.

Cooccurrence module
-----------------------

.. automodule:: src.Cooccurrence
   :members:
   :undoc-members:
   :show-inheritance:

Example
-----------------------
In the following code snippet we have a corpus of one document, 
and we apply a word-occurence transformation with params: 
graph type as Digraph, window_size of 1, English language, adjacency matrix as desired output format, etc::
   
   from text2graphapi.src.Cooccurrence import Cooccurrence
   
   corpus_docs = [{'id': 1, 'doc': 'The violence on the TV. The article discussed the idea of the amount of violence on the news'}]

   to_cooccurrence = Cooccurrence(
                  graph_type = 'DiGraph', 
                  apply_preprocessing = True, 
                  parallel_exec = False,
                  window_size = 1, 
                  language = 'en',
                  output_format = 'adj_matrix')
                  
   output_text_graphs = to_cooccurrence.transform(corpus_docs)
 
After the execution of this code, we have one directed graph with 8 nodes and 15 edges::

   [{
      'doc_id': 1, 
      'graph': <8x8 sparse array of type '<class 'numpy.int64'>' Sparse Row format>, 
      'number_of_edges': 15, 
      'number_of_nodes': 8, 
      'status': 'success'
   }]
