import pandas as pd
from tqdm import tqdm
from preprocess_booklets import preprocess_booklets
from embed_booklets import embed_booklets
from prompts.prompts import prompts
from model import CodeHealersModel
from utils import calc_rouge_score

train_df = pd.read_csv('./data/data/Train.csv')


df_booklet = preprocess_booklets()
vector_store = embed_booklets(df=df_booklet)

retriever = vector_store.as_retriever(search_kwargs=dict(k=5))

model = CodeHealersModel(retriever=retriever, prompt_string=prompts["basic_prompt_1"])

for index, row in (train_df[0:1].iterrows()):
    
    question = row['Question Text']
    actual = row['Question Answer']
    # model_output = model.generate(question=question)
    answer = model.get_answer(question=question)
    context = model.get_context_booklets()
    print(question)
    print('---------------')
    print(answer)
    print('---------------')
    print(context)
    print('---------------')

    rouge_scores = calc_rouge_score(pred=answer, actual=actual)
    print(rouge_scores)
    # print(answer_raw['answer'])
    # print('---------------')
    # print(answer_raw['context'])
    # print('---------------')
