from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

def build_retriever(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in docs]  # Extract strings from Document objects
    vectorstore = FAISS.from_texts(texts, embeddings)  # Pass strings here
    return vectorstore.as_retriever()
