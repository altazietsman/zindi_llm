import pandas as pd
import json

import os
from dotenv import load_dotenv

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import (
#     OpenAIEmbeddings,
#     HuggingFaceInstructEmbeddings,
# )
from langchain_community.vectorstores import FAISS

# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import (
    # PromptTemplate,
    ChatPromptTemplate,
    # HumanMessagePromptTemplate,
    # SystemMessagePromptTemplate,
)

# from utils.embeddings import Embedder
import numpy as np
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def load_dataset(dataset_name: str = "booklet_clean.csv") -> pd.DataFrame:
    """
    Load dataset from file_path

    Args:
        dataset_name (str, optional): Dataset name. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: Dataset
    """

    data_dir = "./data/data/resources"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df


def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> list:
    """
    Create chunks from the dataset

    Args:
        dataset (pd.DataFrame): Dataset
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap

    Returns:
        list: List of chunks
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="cleanText"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )

    return text_chunks


def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get vector store

    Args:
        chunks (list): List of chunks

    Returns:
        FAISS: Vector store
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    if not os.path.exists("./db"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    df = load_dataset()
    chunks = create_chunks(df, 1000, 0)

    # if "vector_store" not in st.session_state:
    vector_store = create_or_get_vector_store(chunks)

    retriever = vector_store.as_retriever(search_kwargs=dict(k=5))

    prompt_string = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        # "I also want to you extract the top 5 keywords from the context. "
        # 'Output the answer in JSON using "Answer": <Answer>, "Keywords": <Keywords>. '
        "\nQuestion: {question} \nContext: {context}"
    )

    # rag_prompt = QA_PROMPT
    rag_prompt = ChatPromptTemplate.from_template(prompt_string)

    rag_prompt = hub.pull("rlm/rag-prompt")
    # print(rag_prompt)

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="TheBloke/Llama-2-7B-Chat-GGUF",
    #     task="text-generation",
    #     # pipeline_kwargs={"max_new_tokens": 10},
    # )

    llm = LlamaCpp(
        model_path="/Users/brendentaylor/Documents/zindi_llm/llama-2-7b-chat.Q6_K.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
    )

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    question = "Compare the laboratory confirmation methods for Chikungunya and diabetes, and which diseases are diagnosed through blood glucose measurements?"

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    temp = rag_chain_with_source.invoke(question)
    # temp = qa_chain.invoke(question)
    return temp


if __name__ == "__main__":
    temp_raw = main()
    # print(temp_raw)
    # print(type(temp_raw))
    # print(type(temp_raw['context']))
    # print('------------------')
    # print(temp_raw['question'])
    # print('------------------')
    # print(temp_raw['answer'])
    # print('------------------')
    # temp = json.loads(main())
    # print(temp)
