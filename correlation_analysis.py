import pandas as pd
from bert_embedder import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity
from numpy import interp
from tqdm import tqdm


def main(data_path='data/join_scores.csv'):
    data = pd.read_csv(data_path)

    bert = BertEmbedder('models/bert-base-cased-pt-br')
    itd_bert = BertEmbedder('models/itd_bert')

    rb = []
    rib = []

    for idx, row in tqdm(data.iterrows()):
        sa_b = bert.get_embeddings(row.sentence_A)[0]
        sb_b = bert.get_embeddings(row.sentence_B)[0]
        sim_b = cosine_similarity([sa_b.numpy()], [sb_b.numpy()])[0][0]
        rb.append(interp(sim_b, [0, 1], [0, 4]))

        sa_ib = itd_bert.get_embeddings(row.sentence_A)[0]
        sb_ib = itd_bert.get_embeddings(row.sentence_B)[0]
        sim_ib = cosine_similarity([sa_ib.numpy()], [sb_ib.numpy()])[0][0]
        rib.append(interp(sim_ib, [0, 1], [0, 4]))

    data["score_BERT"] = rb
    data["score_ITD_BERT"] = rib

    columns = ['score_specialist', 'score_heuristic', 'score_tfidf', 'score_balanced',
               'score_BERT', 'score_ITD_BERT']
    print("PEARSON:\n", data[columns].corr(method='pearson'))
    print("\nSPEARMAN:\n", data[columns].corr(method='spearman'))

    data.to_csv('data/all_scores.csv', index=False)


if __name__ == "__main__":
    main()
