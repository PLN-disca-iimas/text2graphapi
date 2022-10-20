import re  
import emoji
import nltk
import string
import codecs
import contractions
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor  


stopwords = []
for line in codecs.open('stopwords_english.txt', encoding = "utf-8"):
    # Remove black space if they exist
    stopwords.append(line.strip())
stopwords = dict.fromkeys(stopwords, True)   


class Preprocessing(object):
    """Text parser for the preprocessing.

    Params:
        pos_tagger: Tagger for part of speech.    
    
    Examples:
        >>> text = "I am an 🤖 hehe :-)). Lets try :D another one 😲. It seems 👌"
        >>> pre = Preprocessing()
        >>> pre.handle_emoticons(text)
        I am an robot face hehe Very happy. Lets try Laughing, big grin or laugh with glasses
         another one astonished. It seems ok hand
    """

    def __init__(self, pos_tagger=nltk.pos_tag):
        self.pos_tagger = pos_tagger

    def handle_blank_spaces(self, text: str) -> str:
        """Remove blank spaces.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without blank space.
        """    
        return re.sub(r'\s+', ' ', text).strip()

    def handle_non_ascii(self, text: str) -> str:
        """Remove special characters.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without non_asccii characters.        
        """
        regex_non_asccii = f'[^{string.ascii_letters}]'
        return re.sub(regex_non_asccii, " ", text)
    
    
    def handle_emoticons(self, text: str) -> str:
        """Transform emoji to text.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text with emoji text.
        """
        ## Join and clen emojis and emoticons
        all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS}
        all_emoji_emoticons = {k:v.replace(":","").replace("_"," ").strip() 
                                for k,v in all_emoji_emoticons.items()}
        
        # Add emojis for remplace
        kp_all_emoji_emoticons = KeywordProcessor()
        for k,v in all_emoji_emoticons.items():
            kp_all_emoji_emoticons.add_keyword(k, v)

        return kp_all_emoji_emoticons.replace_keywords(text)
        
    def handle_html_tags(self, text: str) -> str:
        """Remove any html tags.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without tags.
        """
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)  

    def handle_stop_words(self, text: str) -> str:
        """Remove stop words
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without stopwords.        
        """
        tokens = self.word_tokenize(text)
        # Remove las stopwords
        without_stopwords = [word for word in tokens if not stopwords.get(word.lower().strip(), False)]
        return " ".join(without_stopwords)
    
    def handle_contractions(self, text: str) -> str:
        """Expand contractions.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without contractions.   
        """
        expanded_words = [contractions.fix(word) for word in text.split(" ")]
        return " ".join(expanded_words)  
        
    def handle_negations(self, text: str) -> str:
        """Handle negations.        

        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text without negations.   
        """
        return self.handle_contractions(text)

    def to_lowercase(self, text: str) -> str:
        """Tranform text to lowercase.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text in lowercase.   
        """
        return text.lower()

    def sent_tokenize(self, text: str) -> list:
        """Tokenize by sentece.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text tokenize by sentences.  
        """
        return nltk.sent_tokenize(text)

    def word_tokenize(self, text: str) -> list:
        """Tokenize by word.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text tokenize by word.  
        """
        return nltk.word_tokenize(text)

    def pos_tagger(self, text: list) -> list:
        """Tagging part of speech.
        
        Params:
            text (str): Text for preprocesesing.
        Return:
            str: Text tagged.         
        """
        return self.pos_tagger(text)
