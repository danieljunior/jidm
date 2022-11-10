import pandas as pd

from tfidf_embedder import TfIdfEmbedder
from preprocess_text import preprocess_texts


def train(to_anotate_path='data/to_anotate.csv', result_path='models/tfidf/to_anotate.bin'):
    data = pd.read_csv(to_anotate_path)
    texts = list(set(data.sentence_A.tolist() + data.sentence_B.tolist()))
    preprocessed_texts = preprocess_texts(texts)
    tfidf_embedder = TfIdfEmbedder()
    tfidf_embedder.train(preprocessed_texts)
    tfidf_embedder.save(result_path)


if __name__ == "__main__":
    train()
