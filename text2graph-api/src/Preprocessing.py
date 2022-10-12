import nltk

class Preprocessing(object):
    def __init__(self, pos_tagger=nltk.pos_tag):
        self.pos_tagger = pos_tagger

    # remove blank spaces
    def handle_blank_spaces(self, text: str) -> str:
        pass

    # replace/remove special char
    def handle_non_ascii(self, text: str) -> str:
        pass

    # transform emoji to text
    def handle_emoticons(self, text: str) -> str:
        pass
    
    # remove html tags
    def handle_html_tags(self, text: str) -> str:
        pass

    # remove stop words
    def handle_stop_words(self, text: str) -> str:
        pass
    
    # expand contractions
    def handle_contractions(self, text: str) -> str:
        pass
    
    # handle negations
    def handle_negations(self, text: str) -> str:
        pass

    # tranform text to lowercase
    def to_lowercase(self, text: str) -> str:
        pass

    # sentence tokenizer
    def sent_tokenize(self, text: str) -> list:
        pass

    # work tokenizer
    def word_tokenize(self, text: str) -> list:
        pass

    # part of speech tags
    def pos_tagger(self, text: list) -> list:
        pass
