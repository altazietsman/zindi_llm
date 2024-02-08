import numpy as np
import os
import pandas as pd
import pathlib



def search_content(query, df_sentances, index, embedder, k=5):
    """Function used to to returns relevant text based on query
    
    Arguments:
    ----------
    query: str
            query text

    df_sentances: pandas dataframe
                  data frame with text columns that match index

    index: faiss index
           index of text embeddings
    
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
    matched_em, matched_indexes = index.search(query_vector, k)
    ids = matched_indexes[0][0:k]

    df = df_sentances.iloc[ids.tolist() ]

    return df


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
            df_booklet = pd.read_excel(dir_path + file, engine='openpyxl')
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

def rename_booklets(dir_path:str):
    """Rename booklets to 'booklet<i>

    Arguments:
    ----------
    dir_path: str
              path of where booklets are stored

    Return:
    -------
    None
    """

    files = os.listdir(pathlib.Path(dir_path))

    for file in files:
        if file.startswith("TG Booklet"):
            first_digit = file[11]
            os.rename(dir_path + file, dir_path + "booklet" + first_digit + ".xlsx")
            # os.rename(dir_path + file, dir_path + file[:8] + ".xlsx")
        elif file.startswith("booklet"):
            print('Seems like the booklets have already been renamed...')
            return None

    return None
