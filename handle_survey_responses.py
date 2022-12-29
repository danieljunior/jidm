import os
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    path = "data/respostas_especialistas"
    survey_responses_df = []
    for file in os.listdir(path):
        survey_responses_df.append(handle_survey_responses(path + '/' + file))

    results = pd.concat(survey_responses_df)
    results.to_csv('data/surveys_responses.csv', index=False)
    # responses_df.to_csv('data/responses_dataframe.csv', index=False)
    # heuristic_df = pd.read_csv('data/to_anotate_heuristic_balanced.csv')
    # join_scores = join_responses_heuristic(responses_df, heuristic_df)
    # join_scores.to_csv('data/join_scores.csv', index=False)
    #
    # pearson, spearman = calculate_correlations(join_scores)
    # print("PEARSON:\n", pearson)
    # print("SPEARMAN:\n", spearman)
    # generate_score_chart(join_scores)


def handle_survey_responses(responses_path):
    responses = pd.read_csv(responses_path, header=None, index_col=False)
    responses.drop(columns=responses.columns[0], axis=1, inplace=True)
    guide_utility = []
    texts_and_similarity = []
    similarity_confidence = []
    text1_highlight = []
    text1_highlight_position = []
    text2_highlight = []
    text2_highlight_position = []
    for column in tqdm(responses):
        column_question = column % 6
        if len(responses.columns) == column:
            guide_utility.append([int(x[1]) for x in responses[column].items() if x[0] > 0])
        elif column_question == 1:
            texts_and_similarity.append(handle_similarity_column(responses[column]))
        elif column_question == 2:
            similarity_confidence.append([int(x[1]) for x in responses[column].items() if x[0] > 0])
        elif column_question == 3:
            text1_highlight.append([x[1] for x in responses[column].items() if x[0] > 0])
        elif column_question == 4:
            text1_highlight_position.append([x[1] for x in responses[column].items() if x[0] > 0])
        elif column_question == 5:
            text2_highlight.append([x[1] for x in responses[column].items() if x[0] > 0])
        elif column_question == 0:
            text2_highlight_position.append([x[1] for x in responses[column].items() if x[0] > 0])
    data = []
    for idx, item in enumerate(flatten(texts_and_similarity)):
        tmp = []
        tmp.append(item[0])
        tmp.append(item[1])
        tmp.append(item[2])
        tmp.append(flatten(similarity_confidence)[idx])
        tmp.append(flatten(text1_highlight)[idx])
        tmp.append(flatten(text1_highlight_position)[idx])
        tmp.append(flatten(text2_highlight)[idx])
        tmp.append(flatten(text2_highlight_position)[idx])
        data.append(tmp)

    responses_df = pd.DataFrame(data, columns=['TEXT1', 'TEXT2',
                                               'SIMILARITY_SCORE', 'SIMILARITY_CONFIDENCE',
                                               'TEXT1_HIGHLIGHT', 'TEXT1_HIGHLIGHT_POSITION',
                                               'TEXT2_HIGHLIGHT', 'TEXT2_HIGHLIGHT_POSITION'])

    return responses_df


def handle_similarity_column(data):
    t1, t2 = None, None
    resp = []
    for index, value in data.items():
        if index == 0:
            t1 = value.split("𝗧𝗘𝗫𝗧𝗢 𝟮:")[0].replace("𝗧𝗘𝗫𝗧𝗢 𝟭:\n\n", "").strip()
            t2 = value.split("𝗧𝗘𝗫𝗧𝗢 𝟮:")[1].replace("𝗧𝗘𝗫𝗧𝗢 𝟮:\n\n", "").strip().split(
                "𝗤𝘂ã𝗼 𝘀𝗶𝗺𝗶𝗹𝗮𝗿𝗲𝘀")[0].strip()
        else:
            score = int(value.split("-")[0].strip())
            resp.append([t1, t2, score])
    return resp


def flatten(list_):
    return [x for x in itertools.chain(*list_)]


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
