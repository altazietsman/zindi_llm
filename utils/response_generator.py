
import pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch


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