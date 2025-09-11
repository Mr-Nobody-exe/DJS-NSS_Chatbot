from modules.loader import load_data
from modules.retriever import build_retriever
from modules.prompt_engineering import qa_prompt
from modules.llm_wrapper import generate_response

class NSSChatbot:
    def __init__(self):
        docs = load_data()
        # Flatten docs into a list of strings
        all_texts = self.flatten_docs(docs)
        self.retriever = build_retriever(all_texts)

    def flatten_docs(self, docs):
        texts = []
        if isinstance(docs, dict):
            for key, value in docs.items():
                texts.extend(self.flatten_docs(value))
        elif isinstance(docs, list):
            for item in docs:
                texts.extend(self.flatten_docs(item))
        else:
            texts.append(str(docs))
        return texts

    def ask(self, query):
        context = self.retriever.invoke(query)
        prompt = qa_prompt.format(context=context, question=query)
        return generate_response(prompt)
