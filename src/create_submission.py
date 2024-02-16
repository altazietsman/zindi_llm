import pandas as pd
from tqdm import tqdm
from preprocess_booklets import preprocess_booklets
from embed_booklets import embed_booklets
from prompts.prompts import prompts
from model import CodeHealersModel
from utils import calc_rouge_score, extract_keyword, get_submission_df, print_line, join_submissions
from langchain import hub
import time
import numpy as np

def create_submission():
    test_df = pd.read_csv('./data/data/Test.csv')

    df_booklet = preprocess_booklets()
    vector_store = embed_booklets(df=df_booklet, recreate_embeddings=False)

    retriever = vector_store.as_retriever(search_kwargs=dict(k=5))

    prompt_string = prompts['hub_rag_prompt']
    print("Prompt:")
    print(prompt_string)
    print_line()

    model = CodeHealersModel(retriever=retriever, prompt_string=prompt_string)

    num_chunks = 8
    test_df_chunks = np.array_split(test_df, num_chunks)
    i=1
    for test_df_chunk in test_df_chunks:
        test_df_chunk = pd.DataFrame(test_df_chunk).reset_index()
        print(f'Starting chunk {i} with length {len(test_df_chunk)}...')
        responses = []
        for index, row in tqdm(test_df_chunk.iterrows()):
            
            response = {}

            question = row['Question Text']
            answer = model.get_answer(question=question)
            print("Question: "+question)
            print("Answer: "+answer)
            print_line()

            keywords = extract_keyword(answer, top_n=5)

            response['keywords'] = keywords
            response['answer'] = answer
            response['reference_document'] = 'book'
            response['paragraph(s)_number'] = 1

            responses.append(response)

        df_responses = pd.DataFrame(responses)

        df_responses['ID'] = test_df_chunk['ID']
        df_responses['Question'] = test_df_chunk['Question Text']

        df_submission_chunk = get_submission_df(df_responses=df_responses)

        # df_responses.columns = ['keywords', 'question_answer', 'reference_document', 'paragraph(s)_number', 'ID', 'Question']

        # df_submission = pd.melt(df_responses, id_vars=['ID'], value_vars=['question_answer', 'reference_document', 'paragraph(s)_number', "keywords"])
        # df_submission['ID'] = df_submission['ID'] + '_' + df_submission['variable']
        # df_submission.columns = ["ID", "variable", "Target"]
        # df_submission = df_submission[['ID', "Target"]].set_index("ID")

        df_submission_chunk.to_csv(f'./data/submissions/submission_bt_sample_{i}.csv', index=True)
        if i < num_chunks:
            i+=1
            print('Resting for a little :) ...')
            time.sleep(240)

    df_submission = join_submissions()
    df_submission.to_csv(f'./data/submissions/submission_bt.csv', index=True)

if __name__ == "__main__":
    create_submission()

