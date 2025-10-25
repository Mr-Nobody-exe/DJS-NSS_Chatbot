from langchain.prompts import PromptTemplate

def create_prompt():
    template = """
You are NSSBot, the official assistant for DJS NSS.
Use the retrieved NSS data below to answer questions.
If you cannot find an answer in the context, say "I’m not sure, please contact the NSS coordinator."

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)
