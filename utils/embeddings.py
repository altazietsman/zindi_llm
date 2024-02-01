from transformers import AutoModel, AutoTokenizer
import torch as pt 
import numpy as np
import torch

class Embedder:

    def __init__(self, model_name:str):

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # add padding token since not all models have it
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(tokenizer))

    def mean_pooling(self, model_output, attention_mask):

        """Perform mean pooling on token embeddings base d on an attention mask
            
        This method takes the output of a language model and an attention mask to 
        compute a mean-pooled representation of the input tokens.
            
        Arguments:
        ----------
        model_output: torch.Tensor
                          The output from a language model, where the first element contains token embeddings for the input text
        attention_mask: torch.Tensor
                        A binary mask indicating which tokens should be considered in the poolinh.
                        It should have the same length as the input text, with 1s for tokens to include and 0s for tokens to exclude
                            
        Returns:
        --------
        embeddings: numpy array
                    The mean-pooled representation of token embeddings, where each token's embedding is weighted by the attention mask
        """

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) /  torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.numpy()

    def embed(self, X, batch_size=32, normalize_embeddings=True, convert_to_tensor=False):

        """Embeds text
            
        Arguments:
        ---------
        X: iterable[str]
           text to embed
        batch_size: int
                    batch size
            
        Returns:
        --------
        embeddings: numpy array
                    embeddings
        """

        if isinstance(X, str):
            X = [X]

        text_embeddings = []

        for i in range(0, len(list(X)), batch_size):
            text_batch = list(X)[i: i + batch_size]
            encoded_input = self.tokenizer(text_batch, padding=True, return_tensors="pt")

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            text_embeddings.append(embeddings)
            
        embeddings = np.vstack(text_embeddings)

        if normalize_embeddings:
            #L2 norm
            norm = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

            embeddings = embeddings / norm
            
        if convert_to_tensor:
            embeddings = pt.from_numpy(embeddings)

        if len(X) == 1:
            embeddings = embeddings[0]
            
        return embeddings