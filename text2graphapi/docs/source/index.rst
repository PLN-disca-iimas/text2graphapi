.. text2graph documentation master file, created by
   sphinx-quickstart on Wed Mar 29 22:00:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. _home:

Welcome to text2graph's!
======================================

**text2graphapi** is a python library for text-to-graph tranformations. 
To use this library it is necessary to install the modules and dependencies in user’s application. 

.. Also, the corpus of text documents to be transformed into graphs has to be loaded and read.


Graph Transformations
==================
Currently, this library support two types of graph representation: 

* :ref:`Cooccurrence`
* :ref:`Heterogeneous`


Overview
==================
The  text to graph transformation pipeline consists of three main modules:

- **Text Preprocessing and Normalization**. This module aims to perform all the cleaning and pre-processing part of the text. Handling blank spaces, emoticons, HTML tags stop words, etc
- **Graph Model**. This module aims to define the entities/nodes and their relationships/edges according to the problem specification. 
- **Graph Transformation**. This module aims to apply vector transformations to the graph as final output, such as adjacency matrix, dense matrix, etc.

The following diagram depicts the pipeline overview for the text to graph tranformation described above:

.. image:: https://www.linkpicture.com/q/texto-to-graph.pipeline.png#center


**Installation from PYPI**
Inside your project, from your CLI type the following command in order to install the latest version of the library::

   pip install text2graphapi