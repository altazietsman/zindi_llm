
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoTokenizer, BitsAndBytesConfig
import yake
from utils.utils import search_content
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class ResponseGenerator:
    """Response Generator loads a model and generates a response.
    
    Attributes:
    -----------
    model: AutoModelForCausalLM
           LLM model
    tokenizer: AutoTokenizer
               tokenizer
    """

    def __init__(self, model_name:str, quantized=False):
        """Initialize class"""
        torch.set_default_device("cpu")
        if quantized:
            self.model = self.load_quantized_model(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, query:str, max_length=200):
        """Generates text response
        
        Arguments:
        ----------
        query: str
               question
        max_lenght: int
                    length of response
        
        Returns:
        --------
        response: str
                  text response
        """

        inputs = self.model.generate(query)
        outputs = self.tokenizer(**inputs, max_length=max_length)
        response = self.tokenizer.batch_decode(outputs)[0]
        return response

    def load_quantized_model(self, model_name: str):
        """Load quantized model

        Arguments:
        ---------
        model_name: str
                    name of model to load

        Return:
        -------
        model: AutoModelForCausalLM
               loaded quantized model
        """

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        return model
        

def get_response(text, llm, booklet_matches, text_column,  gpu=False):
    """Generate response using 
    
    Arguments:
    ----------
    text: str
          query
    llm: model
         loaded model to use
    text_column: str
                 column name of text
    booklet_matches: list
                    book text matches found

    Return:
    -------
    response_dict: dict
                   response information
    """

    paragraph_words = []

    for paragraph in booklet_matches:
        paragraph_words += paragraph.split(" ")
        
    booklet_information = " ".join(paragraph_words)

    try:
        query=f"""You are a specialist in Malawian public health. 
        You have access to the context below, and must answer the question posed to you based entirely on that context. 
        Keep your answer to a maximum of 2 sentence, and use proper grammar.
        If you don't know the answer, say that you don't know. Don't make up an answer.
    \nQuestion: {text} \nContext: {booklet_information}"""                   
        response = llm.generate(query)

    except:

        booklet_information = " ".join(paragraph_words[:250])

        query=f"""You are a specialist in Malawian public health. 
        You have access to the context below, and must answer the question posed to you based entirely on that context. 
        Keep your answer to a maximum of 2 sentence, and use proper grammar.
        If you don't know the answer, say that you don't know. Don't make up an answer.
    \nQuestion: {text} \nContext: {booklet_information}""" 
        response = llm.generate(text)

    clean_sentances = []
    for sentance in response.split("\n"):
        clean_sentances.append(sentance.strip())

    response = " ".join(clean_sentances)


    response_dict = {"answer": response}

    return response_dict

def get_paragraph_info(query, df_booklet, embedder, fastIndex):
    """Return book number and paragraph numbers of matches
    
    Arguments:
    ----------
    query: str
           question
    df_booklet : pandas dataframe
                 original booklet 
    embedder: embedding model
    fastIndex: fastIndex
               index of book
     
    Return:
    -------
    dict: book and paragraph info
    """

    df_book_matches = search_content(query=query, df_sentances=df_booklet, book_index=fastIndex, embedder=embedder, k=3)

    # get matched book
    book = df_book_matches['book'].mode()

    if not book.empty:
        book = book.iloc[0]

    max_pages = df_book_matches[df_book_matches['book'] == book]["index"].max() 
    min_pages = df_book_matches[df_book_matches['book'] == book]["index"].min()

    if (max_pages != min_pages):
        return {"book":book, "paragraphs": f"{min_pages}-{max_pages}"}
    else:
        return {"book":book, "paragraphs": f"{min_pages}"}


def extract_keyword(text, top_n=4):
    """Extract Keywords from text. To install: pip install git+https://github.com/LIAAD/yake
    
    Arguments:
    ----------
    text: str
          text from which to extract keywords
    top_n: int
           number of keywords
    
    Return:
    -------
    keywords: list[str]
              list of keywords
    """

    kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=None)
    keywords_predictions = kw_extractor.extract_keywords(text)
    keywords = [prediction[0] for prediction in keywords_predictions]
    
    return keywords

def find_matching_paragraphs(text_to_check, df_booklet, threshold=0.9):
    """Returns matching paragraphs
    
    Arguments:
    ----------
    text_to_check: str
    df_booklet: pandas_dataframe
                dataframe with booklet information
    threshold: float
                threshold to find matches

    Return:
    ------
    dict: dictionary with book and paragraph information
    """

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the text in the DataFrame
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_booklet['cleanText'])

    # Transform the provided text
    provided_text_tfidf = tfidf_vectorizer.transform([text_to_check])

    # Calculate cosine similarity between the provided text and each paragraph in the DataFrame
    cosine_similarities = cosine_similarity(provided_text_tfidf, tfidf_matrix).flatten()

    # Find paragraphs that meet or exceed the threshold
    matching_paragraph_indices = [i for i, score in enumerate(cosine_similarities) if score >= threshold]

    if matching_paragraph_indices:
        # Get the corresponding paragraph numbers
        matching_paragraph_numbers = df_booklet.iloc[matching_paragraph_indices]['paragraph'].tolist()
        matching_paragraph_numbers = [str(int(i)) for i in matching_paragraph_numbers]
        matching_book_number = df_booklet.iloc[matching_paragraph_numbers]['book'].values.tolist()
        matching_book_number = [f"TG Booklet {i[-1]}" for i in matching_book_number]
        return {"book": ", ".join(matching_book_number), "paragraph":', '.join(matching_paragraph_numbers)}
    
    else:
        # If no paragraphs meet the threshold, fallback to selecting the paragraph with the highest similarity
        closest_paragraph_index = cosine_similarities.argmax()
        closest_paragraph_number = df_booklet.iloc[closest_paragraph_index]['paragraph']
        closest_book_number = df_booklet.iloc[closest_paragraph_index]['book']
        return {"book": f"TG Booklet {closest_book_number[-1]}" , "paragraph":', '.join([str(closest_paragraph_number)])}  # Return as a list