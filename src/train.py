import pandas as pd
import random
from tqdm import tqdm
from preprocess_booklets import preprocess_booklets
from embed_booklets import embed_booklets
from prompts.prompts import prompts
from model import CodeHealersModel
from utils import calc_rouge_score, extract_keyword, get_submission_df, print_line
from langchain import hub

def train():
    train_df = pd.read_csv('./data/data/Train.csv', encoding='utf8')

    df_booklet = preprocess_booklets()
    vector_store = embed_booklets(df=df_booklet, recreate_embeddings=False)

    retriever = vector_store.as_retriever(search_kwargs=dict(k=5))

    prompt_string = prompts['hub_rag_prompt']
    print("Prompt:")
    print(prompt_string)
    print_line()

    model = CodeHealersModel(retriever=retriever, prompt_string=prompt_string)

    responses = []
    train_df = train_df.sample(n=20)
    for index, row in tqdm(train_df.iterrows()):

        response = {}
        
        question = row['Question Text']
        actual = row['Question Answer']
        answer = model.get_answer(question=question)
        context = model.get_context_booklets()

        keywords = extract_keyword(answer, top_n=5)

        print("Question: "+question)
        print("Answer: "+answer)
        print("Keywords:")
        print(*keywords, sep=', ')
        print("Context: ")
        print(*context, sep = ", ")
        print_line()

        response['keywords'] = keywords
        response['answer'] = answer
        response['actual_answer'] = actual

        responses.append(response)


    df_responses = pd.DataFrame(responses)
    df_responses['ID'] = train_df['ID']
    df_responses['Question'] = train_df['Question Text']
    df_responses.columns = ['Keywords', 'Answer', 'actual_answer', 'ID', 'Question']
    df_responses = calc_rouge_score(df=df_responses)

    df_responses.to_csv('./data/train_responses.csv', index=False)

    print(df_responses.loc[:, 'rouge_score'].mean())


if __name__ == '__main__':
    train()
