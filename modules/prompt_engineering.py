from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are the DJS NSS chatbot. 
    Answer only from the given context. 
    If you don’t know, say “Please check with NSS coordinators.”

    Context: {context}
    Question: {question}
    Answer:
    """
)
