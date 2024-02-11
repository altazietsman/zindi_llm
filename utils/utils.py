import numpy as np
import os
import pandas as pd
import pathlib
import re



def search_content(query, df_sentances, df_questions, book_index, question_index, embedder, k=5):
    """Function used to to returns relevant text based on query
    
    Arguments:
    ----------
    query: str
            query text

    df_sentances: pandas dataframe
                  data frame with text columns that match book index

    df_questions: pandas dataframe
                  data frame with text columns that match question index


    book_index: faiss index
                index of booklet embeddings

    question_index: faiss index
                    index of question embeddings
    
    k: int
       top number of matches to return

    embedder: Embedder
              embedding model class
    
    Returns:
    --------
    pandas dataframe: dataframe with text from top matches
    """
    
    query_vector = embedder.embed(query)
    query_vector = np.expand_dims(query_vector, axis=0)

    # We set k to limit the number of vectors we want to return
    matched_em_book, matched_indexes_book = book_index.search(query_vector, k)
    ids_book = matched_indexes_book[0][0:k]

    df_book = df_sentances.iloc[ids_book.tolist() ]

    matched_em_question, matched_indexes_question = question_index.search(query_vector, k=1)
    ids_question = matched_indexes_question[0][0:k]

    df_question = df_questions.iloc[ids_question.tolist() ]

    return df_book, df_question


def read_booklets(dir_path:str):
    """Reads all excel sheets of the booklets
    
    Arguments:
    ----------
    dir_path: str
              path of where booklets are stored
    
    Return:
    -------
    df_booklet: pandas dataframe
                dataframe oof booklet text
    """
    
    files = os.listdir(pathlib.Path(dir_path))

    booklets = []

    for file in files:
        if file.startswith("book"):
            df_booklet = pd.read_excel(dir_path + file)
            df_booklet["book"] = file[:8]
            df_booklet.columns = ['index', 'text', 'book']
            booklets.append(df_booklet)

    df_booklet = pd.concat(booklets)
    df_booklet['text'] = df_booklet['text'].astype('str')

    return df_booklet
    
def retrieve_booklet_text(df_booklet, ids):
     """Retrieve all row with matching ids
     
     Arguments:
     ----------
     df_booklet: pandas df
                 dataframe with booklet information
     ids: list
          ids to return
     
     Return:
     -------
     pandas dataframe: matching rows
     """
     return df_booklet[df_booklet["index"].isin(ids)]

def clean_text(text):
    """Remove all characters besides letters from text
    
    Arguments:
    ----------
    text: str
          text to clean

    Return:
    -------
    text: str
          cleaned text
    """
    
    text = text.replace("\n", "")
    text = re.sub(r'[^A-Za-z ]+', '', text)
    return text