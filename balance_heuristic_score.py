from tqdm import tqdm
from tfidf_embedder import TfIdfEmbedder

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import interp


def get_ementa(text):
    text = ' '.join(text.split())
    sa = ''
    anterior = None
    for s in text.split('.'):
        if s.isupper() or anterior in ['N', 'ART', 'ARTS', 'ART', '(ART']:
            sa += s
            anterior = s.strip().split(' ')[-1]
        else:
            break
    return sa


def main(original_data_path='data/to_anotate.csv', tfidf_path='models/tfidf/to_anotate.bin'):
    tfidf_embedder = TfIdfEmbedder()
    tfidf_embedder.load(tfidf_path)

    data = pd.read_csv(original_data_path)
    score_tfidf = []
    for index, pair in tqdm(data.iterrows()):
        sa = get_ementa(pair['sentence_A'])
        sb = get_ementa(pair['sentence_B'])
        emb = tfidf_embedder.get_embeddings([sa, sb])
        score = cosine_similarity([emb[0]], [emb[1]])[0][0]
        score = interp(score, [0, 1], [0, 4])
        score_tfidf.append(score)

    data_copy = data.copy()
    data_copy.rename({'score': 'score_heuristic'}, axis=1, inplace=True)
    data_copy['score_tfidf'] = score_tfidf
    data_copy['score_balanced'] = data_copy[['score_heuristic', 'score_tfidf']].mean(axis=1)
    data_copy.to_csv('data/to_anotate_heuristic_balanced.csv', index=False)


if __name__ == "__main__":
    main()
