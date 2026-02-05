'''
DataAccessAgent 
load_text_file → I/O responsibility
preprocess_text → cleaning responsibility
get_tokens_from_file → public method Flask will call

'''
import re
import nltk
from nltk.tokenize import word_tokenize


class DataAccessAgent:
    def __init__(self):
        pass

    def load_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_tokens_from_file(self, file_path):
        raw_text = self.load_text_file(file_path)
        clean_text = self.preprocess_text(raw_text)
        tokens = word_tokenize(clean_text)
        tokens = word_tokenize(clean_text)
        return tokens

