import pandas as pd

from preprocess_booklets import preprocess_booklets
from embed_booklets import embed_booklets
from prompts.prompts import prompts
from model import CodeHealersModel


def main():
    df_booklet = preprocess_booklets()
    vector_store = embed_booklets(df=df_booklet)

    retriever = vector_store.as_retriever(search_kwargs=dict(k=5))
    question = "Compare the laboratory confirmation methods for Chikungunya and diabetes, and which diseases are diagnosed through blood glucose measurements?"

    model = CodeHealersModel(
        retriever=retriever, prompt_string=prompts["basic_prompt_1"]
    )

    raw_answer = model.generate(question=question)

    return raw_answer


if __name__ == "__main__":
    temp_raw = main()
    print(temp_raw)
