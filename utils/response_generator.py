
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoTokenizer, BitsAndBytesConfig
import yake



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
        

def get_response(text, llm, df_matches):
    """Generate response using 
    
    Arguments:
    ----------
    text: str
          query
    llm: model
         loaded model to use
    df_matches: pandas datafram
                text matches found

    Return:
    -------
    response_dict: dict
                   response information
    """

    # get matched book
    book = df_matches['book'].mode()

    if not book.empty:
        book = book.iloc[0]

    max_pages = df_matches[df_matches['book'] == book]["index"].max() 
    min_pages = df_matches[df_matches['book'] == book]["index"].min()

    paragraph_words = []

    for paragraph in df_matches['text'].values.tolist():
        paragraph_words += paragraph.split(" ")
        
    booklet_information = " ".join(paragraph_words)

    try:
        query = prompt = f"""
            
            Q: {text}

            Use the information below to answer: 
            
            "{booklet_information} A:" 

            
            """
            
        response = llm.generate(query)

    except:

        booklet_information = " ".join(paragraph_words[:250])

        query = prompt = f"""
            
            Q: {text}

            The information below can be helpful to generate the answer: 
            
            "{booklet_information} A:" 

            
            """

        response = llm.generate(text)

    clean_sentances = []
    for sentance in response.split("\n"):
        clean_sentances.append(sentance.strip())

    response = " ".join(clean_sentances)


    response_dict = {"answer": response, "book": f"TG Booklet {book[-1]}", 
                     "Paragraph": f"{min_pages}-{max_pages}"}

    return response_dict


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
