import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    responses_df = handle_survey_responses('data/anotacoes.csv')
    responses_df.to_csv('data/responses_dataframe.csv', index=False)
    heuristic_df = pd.read_csv('data/to_anotate_heuristic_balanced.csv')
    join_scores = join_responses_heuristic(responses_df, heuristic_df)
    join_scores.to_csv('data/join_scores.csv', index=False)

    pearson, spearman = calculate_correlations(join_scores)
    print("PEARSON:\n", pearson)
    print("SPEARMAN:\n", spearman)
    generate_score_chart(join_scores)


def handle_survey_responses(responses_csv):
    anotacoes = pd.read_csv(responses_csv, header=None, index_col=False)

    data = []
    for column in tqdm(anotacoes):
        docs = anotacoes[column].iloc[0]
        if len(docs.split("𝗧𝗘𝗫𝗧𝗢 𝟮:")) < 2:
            continue
        t1 = docs.split("𝗧𝗘𝗫𝗧𝗢 𝟮:")[0].replace("𝗧𝗘𝗫𝗧𝗢 𝟭:\n\n", "").strip()
        t2 = docs.split("𝗧𝗘𝗫𝗧𝗢 𝟮:")[1].replace("𝗧𝗘𝗫𝗧𝗢 𝟮:\n\n", "").strip()

        for i in range(1, len(anotacoes[column])):
            score = anotacoes[column].iloc[i]
            if type(score) != str:
                continue

            score = score.split("-")[0].strip()
            data.append([t1, t2, int(score)])

    responses_df = pd.DataFrame(data, columns=["sentence_A", "sentence_B", "score_specialist"])
    return responses_df


def join_responses_heuristic(responses_df, heuristic_df):
    join_scores = pd.merge(responses_df, heuristic_df, how='inner', on=['sentence_A', 'sentence_B'])
    return join_scores


def calculate_correlations(join_scores):
    pearson = join_scores[['score_specialist', 'score_heuristic']].corr(method='pearson')
    spearman = join_scores[['score_specialist', 'score_heuristic']].corr(method='spearman')
    return pearson, spearman


def generate_score_chart(join_scores):
    colors = list(sns.color_palette("husl", 5))
    plot_colors = [colors[v] for v in join_scores['score_specialist'].value_counts().keys()]
    join_scores['score_specialist'].value_counts().plot.pie(figsize=(5, 5), colors=plot_colors)
    plt.savefig('data/specialist')
    plot_colors = [colors[v] for v in join_scores['score_heuristic'].value_counts().keys()]
    join_scores['score_heuristic'].value_counts().plot.pie(figsize=(5, 5), colors=plot_colors)
    plt.savefig('data/heuristic')


if __name__ == "__main__":
    main()
