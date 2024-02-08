import faiss

def create_sentance_booklet(df_booklet):
    """Function to create new dataframe of booklet text, where text are split on sentances. The original ids are still retained.
    
    Arguments:
    ----------
    df_booklet: pandas dataframe
                df with columns text and ID
    
    Return:
    -------
    df_sentances: pandas dataframe
                  df with sentances as text
    """

    df_booklet['sentances'] = [paragraph.split(".") for paragraph in df_booklet['text']]
    df_sentances = df_booklet.drop(["text", "textLength"], axis=1).explode('sentances').reset_index(drop=True)
    
    return df_sentances
    

def create_faise_index(embeddings):
    """Create faiss index from embeddings
    
    Arguments:
    ----------
    embeddings: numpy array
                embeddings 
    
    Return:
    -------
    fastIndex: fais index
               embedding index
    """

    n_dimensions = embeddings.shape[1]
    fastIndex = faiss.IndexFlatL2(n_dimensions)
    fastIndex.add(embeddings.astype('float32'))

    return fastIndex