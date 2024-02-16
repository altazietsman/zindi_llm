from langchain import hub
from utils import hub_prompt_to_string


prompts = {
    "basic_prompt_1": """You are a specialist in Malawian public health. 
    You have access to the context below, and must answer the question posed to you based entirely on that context.
    Keep your answer to a maximum of 2 sentence, and use proper grammar.
    If you don't know the answer, say that you don't know. Don't make up an answer.
    \nQuestion: {question} \nContext: {context}""",
    "hub_rag_prompt": hub_prompt_to_string(hub.pull("rlm/rag-prompt"))
}