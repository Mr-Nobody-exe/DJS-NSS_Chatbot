from modules.retriever import build_retriever as RetrieverBuilder
from modules.llm_wrapper import load_llm
from modules.prompt_engineering import create_nss_prompt
from modules.agent_router import route_query
from modules.memory_manager import get_memory
from modules.response_validator import validate_response
from langchain.chains import LLMChain

class NSSRAGPipeline:
    def __init__(self):
        self.retriever = RetrieverBuilder().build
        self.llm = load_llm()
        self.prompt = create_nss_prompt()
        self.memory = get_memory()

    def answer_query(self, query: str):
        route = route_query(query)
        context_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in context_docs])

        chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
        raw_response = chain.run({"question": query, "context": context})

        validated = validate_response(query, context, raw_response)
        return f"[{route.upper()}] {validated}"
