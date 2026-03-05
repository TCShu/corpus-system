'''
DataAccessAgent 
load_text_file → I/O responsibility
preprocess_text → cleaning responsibility
get_tokens_from_file → public method Flask will call.

'''
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")


class DataAccessAgent:
    def load_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_clean_text_from_file(self, file_path):
        raw_text = self.load_text_file(file_path)
        return self.preprocess_text(raw_text)

    def get_tokens_from_file(self, file_path):
        clean_text = self.get_clean_text_from_file(file_path)
        return word_tokenize(clean_text)