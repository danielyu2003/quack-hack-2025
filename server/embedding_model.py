# embedding_model.py
import fasttext
import fasttext.util
import os

class EmbeddingModel:
    def __init__(self, lang='en'):
        self.lang = lang
        self.model = None
        self.original_dir = os.getcwd()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def __call__(self, x):
        return self.get_sentence_embedding(x)

    def load_model(self):
        os.chdir(self.current_dir)
        try:
            os.makedirs("model", exist_ok=True)
            os.chdir("model/")
            fasttext.util.download_model(self.lang, if_exists='ignore')
            self.model = fasttext.load_model(f'cc.{self.lang}.300.bin')
            fasttext.util.reduce_model(self.model, 100)
        finally:
            os.chdir(self.original_dir)

    def get_sentence_embedding(self, sentence: str):
        words = sentence.split()
        vectors = [self.model.get_word_vector(word) for word in words]
        return sum(vectors) / len(vectors) if vectors else 0.0
