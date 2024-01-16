import os
import re
import sys 
import emoji
import nltk
import string
import codecs
import contractions
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
import spacy
import logging
from nltk.corpus import stopwords
from spacy.cli import download
from spacy.language import Language
from spacy.lang.en import stop_words
from itertools import chain
from spacy.tokens import Doc
import networkx as nx
import networkx

from .configs import DEFAULT_NUM_CPU_JOBLIB


# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, 'src/resources')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
finally:
    from nltk.corpus import wordnet

    

class Preprocessing(object):
    """Text parser for the preprocessing.
    :params pos_tagger: Tagger for part of speech.    

    :examples
        >>> text = "I am an 🤖 hehe :-)). Lets try :D another one 😲. It seems 👌"
        >>> pre = Preprocessing()
        >>> pre.handle_emoticons(text)
        I am an robot face hehe Very happy. Lets try Laughing, big grin or laugh with glasses
         another one astonished. It seems ok hand.
        >>> pre = Preprocessing({'handle_contractions'=True, 'handle_stop_words'=True})

        >>> pre = Preprocessing() # All preprocessing
        >>> pre.make_preprocessing(text)
    """

    def __init__(self, lang='en', steps_preprocessing={}):
        self.lang = lang
        self.param_prepro = steps_preprocessing
        self.methods_preprocessing = {
            'handle_blank_spaces': self.handle_blank_spaces,
            'handle_non_ascii': self.handle_non_ascii,
            'handle_emoticons': self.handle_emoticons,
            'handle_html_tags': self.handle_html_tags,
            #'handle_negations': self.handle_negations,
            'handle_contractions': self.handle_contractions,
            'handle_stop_words': self.handle_stop_words,
            'to_lowercase': self.to_lowercase,
            'handle_blank_spaces': self.handle_blank_spaces
        }

        # Load Spacy model: tokenizer, tagger            
        if self.lang == 'sp':
            stoword_path = RESOURCES_DIR + '/stopwords_spanish.txt'
            self.nlp = self.load_spacy_model("es_core_news_sm")
        elif self.lang == 'fr':
            stoword_path = RESOURCES_DIR + '/stopwords_french.txt'
            self.nlp = self.load_spacy_model("fr_core_news_sm")
        else: #default self.lang == 'en'
            stoword_path = RESOURCES_DIR + '/stopwords_english.txt'
            self.nlp = self.load_spacy_model("en_core_web_sm")
        
        self.nlp.max_length = 10000000000
        #self.nlp.add_pipe("info_component", name="print_info", last=True)
        #self.nlp.add_pipe("multilevel_lang_features", name="multilevel_lang_features", last=True)
        logger.debug(self.nlp.pipe_names)
        #print("------------> ", self.nlp.pipe_names)
        #Doc.set_extension("preproc_text_tokens", default=None)
        #Doc.set_extension("multilevel_lang_info", default=[])


        stopwords = []
        for line in codecs.open(stoword_path, encoding="utf-8"):
            # Remove black space if they exist
            stopwords.append(line.strip())
        self.stopwords = dict.fromkeys(stopwords, True)
        #self.stopwords = set(stopwords.words('english'))


    def load_spacy_model(self, spacy_model):
        exclude_modules = ["ner", "textcat", "tok2vec"]
        try:
            spacy.load(spacy_model, exclude=exclude_modules)
            logger.info('Has already installed spacy model %s', spacy_model)
        except OSError:
            logger.info("Downloading %s model for the spaCy, this will only happen once", spacy_model)
            download(spacy_model)
        finally:
            return spacy.load(spacy_model, exclude=exclude_modules)


    def prepocessing_pipeline(self, text):
        logger.debug('Aplying Text Preprocessing')
        if len(self.param_prepro) == 0:
            # To do all preprocessing
            for method in self.methods_preprocessing:
                text = self.methods_preprocessing[method](text)
        else:
            for method in self.param_prepro:
                if self.param_prepro[method]:
                    text = self.methods_preprocessing[method](text)
            text = self.handle_blank_spaces(text)
        return text


    def handle_blank_spaces(self, text: str) -> str:
        """Remove blank spaces.

        :params str text: Text for preprocesesing.
        :return str: Text without blank space.
        """
        return re.sub(r'\s+', ' ', text).strip()


    def handle_non_ascii(self, text: str) -> str:
        """Remove special characters.

        :params str text: Text for preprocesesing.
        :return str: Text without non_asccii characters.        
        """
        regex_non_asccii = f'[^{string.ascii_letters}]'
        return re.sub(regex_non_asccii, " ", text)


    def handle_emoticons(self, text: str) -> str:
        """Transform emoji to text.

        :params str text: Text for preprocesesing.
        :return str: Text with emoji text.
        """
        # Join and clen emojis and emoticons
        all_emoji_emoticons = {**EMOTICONS_EMO, **UNICODE_EMOJI_ALIAS}
        all_emoji_emoticons = {k: v.replace(":", "").replace("_", " ").strip()
                               for k, v in all_emoji_emoticons.items()}

        # Add emojis for remplace
        kp_all_emoji_emoticons = KeywordProcessor()
        for k, v in all_emoji_emoticons.items():
            kp_all_emoji_emoticons.add_keyword(k, v)

        return kp_all_emoji_emoticons.replace_keywords(text)


    def handle_html_tags(self, text: str) -> str:
        """Remove any html tags.

        :params str text: Text for preprocesesing.
        :return str: Text without tags.
        """
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)


    def handle_stop_words(self, text: str) -> str:
        """Remove stop words

        :params str text: Text for preprocesesing.
        :return str: Text without stopwords.        
        """
        tokens = self.word_tokenize(text)
        # Remove las stopwords
        #without_stopwords = [word for word in tokens if not self.stopwords.get(word.lower().strip(), False)]
        without_stopwords = [word for word in tokens if not word.lower().strip() in self.stopwords]
        return " ".join(without_stopwords)
    

    def handle_contractions(self, text: str) -> str:
        """Expand contractions.

        :params str text: Text for preprocesesing.
        :return str: Text without contractions.   
        """
        expanded_words = [contractions.fix(word) for word in text.split(" ")]
        return " ".join(expanded_words)


    def handle_negations(self, text: str) -> str:
        """Handle negations.  

        :params str text: Text for preprocesesing.
        :return str: Text without negations.   
        """
        return self.handle_contractions(text)


    def to_lowercase(self, text: str) -> str:
        """Tranform text to lowercase.

        :params str text: Text for preprocesesing.
        :return str: Text in lowercase.   
        """
        return text.lower()


    def sent_tokenize(self, text: str) -> list:
        """Tokenize by sentece.

        :params str text: Text for preprocesesing.
        :return str: Text tokenize by sentences.  
        """
        return nltk.sent_tokenize(text)


    def word_tokenize(self, text: str) -> list:
        """Tokenize by word.

        :params str text: Text for preprocesesing.
        :return str: Text tokenize by word.  
        """
        #doc = self.nlp(text)
        #return [str(token.lemma_) for token in doc]
        return nltk.word_tokenize(text)


    def pos_tagger(self, text: str) -> list:
        """Tagging part of speech.

        :params str text: Text for preprocesesing.
        :return str: Text tagged.         
        """
        #doc = self.nlp(text)
        #return [(token.lemma_, token.pos_) for token in doc]
        return nltk.pos_tag(text)

    
    def get_multilevel_lang_features(self, doc) -> list:
        """Get multilevel lang features from text documents (lexical, morpholocial, syntactic and semantic level).

        :params str text: Text for preprocesesing.
        :return str: Text with multilevel lang features.
        """
        doc_tokens = [] 
        for token in doc:
            synonyms_token = wordnet.synsets(str(token.lemma_))
            synonyms_token_head = wordnet.synsets(str(token.head.lemma_))
            synonyms_token_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_token])))
            synonyms_token_head_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_token_head])))
            token_info = {
                'token': token.text,
                'token_lemma': token.lemma_,
                'token_pos': token.pos_,
                'token_dependency': token.dep_,
                'token_head': token.head,
                'token_head_lemma': token.head.lemma_,
                'token_head_pos': token.head.pos_,
                'token_synonyms': synonyms_token_list[:5],
                'token_head_synonyms': synonyms_token_head_list[:5],
                'is_root_token': False,
            }
            if token.dep_ == 'ROOT':
                token_info['is_root_token'] = True
            doc_tokens.append(token_info)

        return doc_tokens
     
        
    def nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        int_synt_graph = nx.DiGraph()
        doc_tuples = []
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        #Doc.set_extension("preproc_text_tokens", default=[])
        #Doc.set_extension("doc_graph", default=None)
        #self.nlp.add_pipe("multilevel_lang_features", name="multilevel_lang_features", last=True)

        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=1, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            '''
            preproc_text =  self.word_tokenize(self.prepocessing_pipeline(doc.text))
            doc.set_extension("prep_text_" , default=preproc_text)
            doc._.preproc_text_tokens = preproc_text
            '''
            '''
            nodes = context['_get_entities'](doc._.multilevel_lang_info)
            edges = context['_get_relations'](doc._.multilevel_lang_info)
            graph = context['_build_graph'](nodes, edges)
            doc._.doc_graph = graph            
            int_synt_graph.add_edges_from(graph.edges(data=True))
            int_synt_graph.add_nodes_from(graph.nodes(data=True))
            '''
            doc_tuples.append((doc, context))
        return doc_tuples

    
# TESTING... *************************************************************
'''
    @Language.component("multilevel_lang_features")
    def multilevel_lang_features(doc) -> list:
        """Get multilevel lang features from text documents (lexical, morpholocial, syntactic and semantic level).

        :params str text: Text for preprocesesing.
        :return str: Text with multilevel lang features.
        """
        doc_tokens = [] 
        for token in doc:
            synonyms_token = wordnet.synsets(str(token.lemma_))
            synonyms_token_head = wordnet.synsets(str(token.head.lemma_))
            synonyms_token_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_token])))
            synonyms_token_head_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_token_head])))
            token_info = {
                'token': token.text,
                'token_lemma': token.lemma_,
                'token_pos': token.pos_,
                'token_dependency': token.dep_,
                'token_head': token.head,
                'token_head_lemma': token.head.lemma_,
                'token_head_pos': token.head.pos_,
                'token_synonyms': synonyms_token_list[:5],
                'token_head_synonyms': synonyms_token_head_list[:5],
                'is_root_token': False,
            }
            if token.dep_ == 'ROOT':
                token_info['is_root_token'] = True
            doc_tokens.append(token_info)

        doc._.multilevel_lang_info = doc_tokens
        return doc


    @Language.component("info_component")
    def info_component(doc):
        print(f"After tokenization, this doc has {len(doc)} tokens.")
        print("The part-of-speech tags are:", [token.pos_ for token in doc])
        if len(doc) < 10:
            print("This is a pretty short document.")
        return doc
    
'''