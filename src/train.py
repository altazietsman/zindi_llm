import pandas as pd
from tqdm import tqdm
from preprocess_booklets import preprocess_booklets
from embed_booklets import embed_booklets
from prompts.prompts import prompts
from model import CodeHealersModel
from utils import calc_rouge_score, extract_keyword, get_submission_df
from langchain import hub

train_df = pd.read_csv('./data/data/Train.csv')


df_booklet = preprocess_booklets()
vector_store = embed_booklets(df=df_booklet, recreate_embeddings=False)

retriever = vector_store.as_retriever(search_kwargs=dict(k=5))

# prompt_string = prompts["basic_prompt_1"]
prompt_string = hub.pull("rlm/rag-prompt")
# print(prompt_string)

model = CodeHealersModel(retriever=retriever, prompt_string=prompt_string)

responses = []
for index, row in tqdm(train_df[0:20].iterrows()):

    response = {}
    
    question = row['Question Text']
    actual = row['Question Answer']
    # model_output = model.generate(question=question)
    answer = model.get_answer(question=question)
    context = model.get_context_booklets()

    keywords = extract_keyword(answer, top_n=5)

    response['keywords'] = keywords
    response['answer'] = answer
    response['actual_answer'] = actual

    responses.append(response)

    # print(question)
    # print('---------------')
    # print(answer)
    # print('---------------')
    # print(keywords)
    # print('---------------')
    # print(context)
    # print('---------------')

    # rouge_scores = calc_rouge_score(pred=answer, actual=actual)
    # print(rouge_scores)


df_responses = pd.DataFrame(responses)
df_responses['ID'] = train_df['ID']
df_responses['Question'] = train_df['Question Text']
df_responses.columns = ['Keywords', 'Answer', 'actual_answer', 'ID', 'Question']
df_responses = calc_rouge_score(df=df_responses)

# print(df_responses.columns)
# print(df_responses.head())

df_responses.to_csv('./data/train_responses.csv', index=False)

