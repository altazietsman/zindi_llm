from llama_cpp import Llama


class llama_cpp:
    """Utilizes the Llama model downloaded at: 
        - Download: `wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_1.bin`
        - Then run: pip install llama-cpp-python==0.1.78
    
    Attributes:
    -----------
    model: lLama
    """

    def __init__(self, model_path:str, gpu=False):

        if gpu:
            self.model = Llama(model_path=model_path, n_gpu_layers=-1,
                                seed=1337, n_ctx=2048)
        else:
            self.model = Llama(model_path=model_path,
                                seed=1337, n_ctx=2048, n_threads=4)

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

        output = self.model(query, echo=False)
        response = output['choices'][0]['text'].strip('\n')
        return response