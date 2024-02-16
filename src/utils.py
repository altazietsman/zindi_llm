from rouge_score import rouge_scorer
import yake
import pandas as pd
import glob
import os

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def calc_rouge_score(df=None, pred=None, actual=None):
    if df is not None and isinstance(df, pd.DataFrame):
        # Apply function to DataFrame
        df['rouge_score'] = df.apply(lambda row: calc_rouge_score(pred=row['Answer'], actual=row['actual_answer']), axis=1)
        return df
    elif pred is not None and actual is not None:
        # Calculate and return the float
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        precision, recall, fmeasure = scorer.score(actual, pred)['rouge1']
        return fmeasure
    else:
        # Handle invalid input
        raise ValueError("Invalid input. Pass either a DataFrame (df) or both pred and actual strings.")


def extract_keyword(text, top_n=4):
    """Extract Keywords from text. To install: pip install git+https://github.com/LIAAD/yake
    
    Arguments:
    ----------
    text: str
          text from which to extract keywords
    top_n: int
           number of keywords
    
    Return:
    -------
    keywords: list[str]
              list of keywords
    """

    kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=None)
    keywords_predictions = kw_extractor.extract_keywords(text)
    keywords = [prediction[0] for prediction in keywords_predictions]
    
    return keywords


def get_submission_df(df_responses):
    # df_responses['ID'] = test_df['ID']
    # df_responses['Question'] = test_df['Question Text']

    df_responses.columns = ['keywords', 'question_answer', 'reference_document', 'paragraph(s)_number', 'ID', 'Question']

    df_submission = pd.melt(df_responses, id_vars=['ID'], value_vars=['question_answer', 'reference_document', 'paragraph(s)_number', "keywords"])
    df_submission['ID'] = df_submission['ID'] + '_' + df_submission['variable']
    df_submission.columns = ["ID", "variable", "Target"]
    df_submission = df_submission[['ID', "Target"]].set_index("ID")

    return df_submission


def hub_prompt_to_string(hub_prompt):
    hub_prompt_string = hub_prompt.dict()['messages'][0]['prompt']['template']
    return hub_prompt_string

def print_line():
    print('-------------------')


def join_submissions(sub_path='./data/submissions'):
    files = glob.glob(f'{sub_path}/submission_bt_sample_*.csv')
    print(f'{len(files)} files found matching pattern. Joining them.')

    booklets = []

    for file in files:
        df_submission = pd.read_csv(file, index_col='ID')
        booklets.append(df_submission)
        os.remove(file)

    df_submission = pd.concat(booklets)
    return df_submission