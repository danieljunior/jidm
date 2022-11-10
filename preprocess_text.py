import ray
import re
import nltk
from NLPyPort.FullPipeline import *

NLPYPORT_OPTIONS = {
    "tokenizer": True,
    "pos_tagger": True,
    "lemmatizer": True,
    "entity_recognition": False,
    "np_chunking": False,
    "pre_load": False,
    "string_or_array": True
}


def preprocess_texts(texts):
    ray.init(num_cpus=4, ignore_reinit_error=True)
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('floresta')
    nltk.download('rslp')
    results = [preprocess.remote(text) for text in texts]
    return [ray.get(i) for i in results]


@ray.remote
def preprocess(text):
    doc = new_full_pipe(text, options=NLPYPORT_OPTIONS)
    tokens = [lema for idx, lema in enumerate(doc.lemas)
              if lema != 'EOS'
              and lema != '']
    tokens = [token for token in tokens
              if token not in nltk.corpus.stopwords.words('portuguese')]

    tokens = [token for token in tokens
              if not re.match('[^A-Za-z0-9]+', token)]

    return ' '.join([token for token in tokens
                     if not any(char.isdigit() for char in token)])
