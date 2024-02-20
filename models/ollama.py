import ollama

class Ollama:

    def __init__(self, model:str, gpu=False):
        """Initilaise the Ollama class
        
        model: str
               name of model to use
        gpu: Bool
        """

        self.model = model
        if gpu:
            self.num_gpu=1
        else:
            self.num_gpu=0

    def generate(self, query:str):
        """Generates response
        
        Arguments:
        ----------
        text: str
              query

        Return:
        -------
        response: str
                  response to query
        """

        response = ollama.generate(model=self.model, prompt=query, options={"num_gpu":self.num_gpu, "seed":1337, "temperature":0})

        return response['response']