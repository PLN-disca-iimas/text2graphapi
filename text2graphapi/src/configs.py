import multiprocessing
import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESOURCES_DIR_PATH = os.path.join(ROOT_DIR, 'src/resources')
OUTPUT_DIR_NAME = 'output_text2graphapi'
OUTPUT_DIR_PATH = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)
OUTPUT_DIR_HETERO_PATH = os.path.join(OUTPUT_DIR_PATH, 'heterogeneous')
OUTPUT_DIR_COOCCUR_PATH = os.path.join(OUTPUT_DIR_PATH, 'coocurrence')

DEFAULT_NUM_CPU_JOBLIB = multiprocessing.cpu_count() // 2
NUM_PRINT_ITER = 1000