from langchain_community.llms import LlamaCpp
import os

from utils import format_docs
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from prompts.prompts import prompts


def set_llm(
    model_path="/Users/brendentaylor/Documents/zindi_llm/llama-2-7b-chat.Q6_K.gguf",
):
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        max_tokens=256,
        top_p=0.5,
        n_gpu_layers=8,
        n_batch=32,
        n_ctx=2048,
        # callback_manager=callback_manager,
        verbose=False,
        seed=os.environ['SEED']
    )

    return llm


def set_model_pipeline(llm, retriever, prompt_str):
    if isinstance(prompt_str, str):
        rag_prompt = ChatPromptTemplate.from_template(prompt_str)
    else: 
        rag_prompt = prompt_str
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


class CodeHealersModel:
    def __init__(self, retriever, prompt_string) -> None:
        self.retriever = retriever
        self.llm = None
        self.rag_chain_with_source = None
        self.prompt_string = prompt_string
        self.llm = set_llm()
        self.chain = set_model_pipeline(
            llm=self.llm, retriever=self.retriever, prompt_str=self.prompt_string
        )
        self.model_output = None

    def generate(self, question):
        model_output = self.chain.invoke(question)
        self.model_output = model_output
        return model_output
    
    def get_answer(self, question):
        self.generate(question=question)

        answer = self.model_output['answer']
        return answer
    
    def get_context_booklets(self):
        if self.model_output is None:
            Warning("Model has not been run yet. Use model.get_answer().")
            return

        contexts = self.model_output['context']

        booklets = []
        indices = []

        for context in contexts:
            context_dic = context.dict()
            # print(context_dic)
            book = context_dic['metadata']['book']
            booklets.append(book)

            index = context_dic['metadata']['index']
            indices.append(index)

        return booklets, indices
        
