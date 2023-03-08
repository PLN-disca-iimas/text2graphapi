import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from math import log
from joblib import Parallel, delayed

from text2graphapi.src.Utils import Utils
from text2graphapi.src.Preprocessing import Preprocessing
from text2graphapi.src.GraphTransformation import GraphTransformation
from text2graphapi.src import Graph

# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneGraph(Graph.Graph):
    def __init__(self):
        ...


    def transform(self):
        ...

