import train_tfidf
import balance_heuristic_score
import handle_survey_responses
import correlation_analysis

if __name__ == "__main__":
    # print('Fitting TFIDF...')
    # train_tfidf.train()
    print('\n\nGenerating balanced score between Heuristic and TFIDF...')
    balance_heuristic_score.main()
    print('\n\nHandle specialist responses and automatic scores...')
    handle_survey_responses.main()
    print('\n\nGenerating correlation analysis between scoring strategies...')
    correlation_analysis.main()
    print('\n\nFinish!')