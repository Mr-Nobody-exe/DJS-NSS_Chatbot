from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LLM_MODEL
import os
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

def load_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0.4)
