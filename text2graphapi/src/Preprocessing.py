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

# Logging configs
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, 'src/resources')

try:
    spacy.load('en_core_web_sm')
    spacy.load('es_core_news_md')
    spacy.load('fr_core_news_sm')
    nltk.data.find('tokenizers/punkt')
    logger.info('Has already installed spacy models')
except OSError:
    logger.info(
        "Downloading language model for the spaCy, this will only happen once")
    from spacy.cli import download
    download('en_core_web_sm')
    download('es_core_news_md')
    download('fr_core_news_sm')
    nltk.download('punkt')



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
        # , pos_tagger=nltk.pos_tag
        # self.pos_tagger = pos_tagger
        stopwords = []
        self.lang = lang
        self.param_prepro = steps_preprocessing
        self.methods_preprocessing = {'to_lowercase': self.to_lowercase,
                                      'handle_negations': self.handle_negations,
                                      'handle_contractions': self.handle_contractions,
                                      'handle_stop_words': self.handle_stop_words,
                                      'handle_html_tags': self.handle_html_tags,
                                      'handle_emoticons': self.handle_emoticons,
                                      'handle_non_ascii': self.handle_non_ascii,
                                      'handle_blank_spaces': self.handle_blank_spaces}

        # Guardamos las stopwords correspondientes
        if self.lang == 'en':
            stoword_path = RESOURCES_DIR + '/stopwords_english.txt'
            # Load English tokenizer, tagger, parser and NER
            self.nlp = spacy.load("en_core_web_sm")
        elif self.lang == 'es':
            stoword_path = RESOURCES_DIR + '/stopwords_spanish.txt'
            # Load Spanish tokenizer, tagger, parser and NER
            self.nlp = spacy.load('es_core_news_md')
        elif self.lang == 'fr':
            stoword_path = RESOURCES_DIR + '/stopwords_french.txt'
            # Load Spanish tokenizer, tagger, parser and NER
            self.nlp = spacy.load('fr_core_news_sm')

        for line in codecs.open(stoword_path, encoding="utf-8"):
            # Remove black space if they exist
            stopwords.append(line.strip())
        self.stopwords = dict.fromkeys(stopwords, True)


    def prepocessing_pipeline(self, text):
        logger.debug('Aplying Text Preprocessing')
        if len(self.param_prepro) == 0:
            # To do all preprocessing
            for method in self.methods_preprocessing:
                self.methods_preprocessing[method](text)
        else:
            for method in self.param_prepro:
                if self.param_prepro[method]:
                    self.methods_preprocessing[method](text)


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
        without_stopwords = [word for word in tokens if not self.stopwords.get(word.lower().strip(), False)]
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
        doc = self.nlp(text)
        return [(token, token.pos_) for token in doc]
