import numpy as np
import _pickle as cPickle

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfEmbedder:
    NLPYPORT_OPTIONS = {
        "tokenizer": True,
        "pos_tagger": True,
        "lemmatizer": True,
        "entity_recognition": False,
        "np_chunking": False,
        "pre_load": False,
        "string_or_array": True
    }

    def __init__(self):
        self.tfidf = TfidfVectorizer()
        # self.ray_initialized = False
        self.cache = None

    def get_embeddings(self, texts):
        texts = [text.lower() for text in texts]
        return self.tfidf.transform(texts).toarray().astype(np.float32)

    def train(self, preprocessed_data):
        self.tfidf.fit(preprocessed_data)

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self.tfidf, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.tfidf = cPickle.load(f)

    def dim(self):
        return self.tfidf.get_feature_names_out().size

    def set_cache(self, cache):
        self.cache = cache
