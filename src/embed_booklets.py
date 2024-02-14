import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def create_chunks(dataset: pd.DataFrame, chunk_size: int=1000, chunk_overlap: int=20) -> list:
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
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )

    return text_chunks


def create_or_get_vector_store(chunks: list, recreate_embeddings=False) -> FAISS:
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
    if not os.path.exists("./db") or recreate_embeddings:
        print("CREATING DB")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("./db")
    else:
        # print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore


def embed_booklets(df, recreate_embeddings=False):
    chunks = create_chunks(df, 1000, 0)
    vector_store = create_or_get_vector_store(chunks, recreate_embeddings=recreate_embeddings)

    return vector_store
