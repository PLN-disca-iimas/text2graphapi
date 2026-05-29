import os
import re
import regex
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

from configs import DEFAULT_NUM_CPU_JOBLIB


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

    

class No_original(object):
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
            'handle_html_tags': self.handle_html_tags,
            'handle_urls': self.handle_urls,
            'handle_emoticons': self.handle_emoticons,
            'to_lowercase': self.to_lowercase,
            'handle_contractions': self.handle_contractions,
            'handle_negations': self.handle_negations,
            'handle_non_ascii': self.handle_non_ascii,
            'handle_blank_spaces': self.handle_blank_spaces,
            'handle_stop_words': self.handle_stop_words
        }

        # Load Spacy model: tokenizer, tagger            
        if self.lang == 'es':
            stoword_path = RESOURCES_DIR + '/stopwords_spanish.txt'
            self.nlp = self.load_spacy_model("es_core_news_sm")

            self.stop_words = spacy.lang.es.stop_words.STOP_WORDS # Lista adicional de stopwords de spacy
        
        elif self.lang == 'fr':
            stoword_path = RESOURCES_DIR + '/stopwords_french.txt'
            self.nlp = self.load_spacy_model("fr_core_news_sm")

            self.stop_words = spacy.lang.fr.stop_words.STOP_WORDS # Lista adicional de stopwords de spacy

        else: #default self.lang == 'en'
            stoword_path = RESOURCES_DIR + '/stopwords_english.txt'
            self.nlp = self.load_spacy_model("en_core_web_sm")

            self.stop_words = spacy.lang.en.stop_words.STOP_WORDS # Lista adicional de stopwords de spacy

        self.nlp.max_length = 10000000 
        
        logger.debug(self.nlp.pipe_names)


        self.stopwords = set()
        for line in codecs.open(stoword_path, encoding="utf-8"):
            # Remove black space if they exist
            self.stopwords.add(line.strip())
        self.stopwords.update(self.stop_words)


    def load_spacy_model(self, spacy_model):
        exclude_modules = []
        try:
            spacy.load(spacy_model, exclude=exclude_modules)
            logger.info('Has already installed spacy model %s', spacy_model)
        except OSError:
            logger.info("Downloading %s model for the spaCy, this will only happen once", spacy_model)
            download(spacy_model)
        
        return spacy.load(spacy_model, exclude=exclude_modules)



    def preprocessing_pipeline(self, text):
        logger.debug('Aplying Text Preprocessing')
        if len(self.param_prepro) == 0:
            # To do all preprocessing
            for method in self.methods_preprocessing:
                text = self.methods_preprocessing[method](text)
        else:
            for method in self.param_prepro:
                if self.param_prepro[method]:
                    text = self.methods_preprocessing[method](text)
        return text


    def handle_blank_spaces(self, text: str) -> str:
        """Remove blank spaces.

        :params str text: Text for preprocesesing.
        :return str: Text without blank space.
        """
        return re.sub(r'\s+', ' ', text).strip()


    def handle_non_ascii(self, text: str) -> str:
        if self.lang == 'en':
            regex_non_ascii = f'[^{string.ascii_letters}]'
            return re.sub(regex_non_ascii, " ", text)
        
        # regex módulo SÍ soporta \p{L}
        regex_keep_letters = r'[^\p{L}\s]'
        text_cleaned = regex.sub(regex_keep_letters, " ", text)
        return regex.sub(r'\s+', ' ', text_cleaned).strip()
        
    def handle_urls(self, text: str) -> str:
        """Maneja URLs en el texto.
        
        :params str text: Texto para preprocesar.
        :return str: Texto con URLs procesadas.
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '[URL]', text)

    def handle_emoticons(self, text: str) -> str:
        """Transform emoji to text.

        :params str text: Text for preprocesesing.
        :return str: Text with emoji text.
        """
        def emoji_to_text(emoji_char, data):
            text_desc = emoji.demojize(emoji_char, language=self.lang)
            text_desc = text_desc.replace("_", " ").replace(":", "").replace("-", " ")
            return text_desc
        
        return emoji.replace_emoji(text, replace=emoji_to_text)


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
        # Remove stopwords
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
        doc = self.nlp(text)
        return [str(token) for token in doc]
        #return nltk.word_tokenize(text)


    def pos_tagger(self, text: str) -> list:
        """Tagging part of speech.

        :params str text: Text for preprocesesing.
        :return str: Text tagged.         
        """
        #doc = self.nlp(text)
        #return [(token.lemma_, token.pos_) for token in doc]
        return nltk.pos_tag(text)
    
    @Language.component("stop_words_component")
    def stop_words_component(self, doc):
        # Do something to the doc here
        for token in doc:
            without_stopwords = [word for word in doc if not word.lower().strip() in self.stopwords]
        return doc

    
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

        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=1, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)

            doc_tuples.append((doc, context))
        return doc_tuples