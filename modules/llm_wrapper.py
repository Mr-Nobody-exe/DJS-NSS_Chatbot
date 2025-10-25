from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LLM_MODEL

def load_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0.4, openai_api_key=OPENAI_API_KEY)
