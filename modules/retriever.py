import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from config import OPENAI_API_KEY, EMBEDDING_MODEL


def flatten_nss_json_for_documents(json_path="nss_data.json"):
    docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all keys
    for section, content in data.items():
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, str):
                    docs.append(Document(page_content=value, metadata={"source": f"{section}.{key}"}))
                elif isinstance(value, list):
                    for item in value:
                        docs.append(Document(page_content=str(item), metadata={"source": section}))
        elif isinstance(content, list):
            for item in content:
                docs.append(Document(page_content=str(item), metadata={"source": section}))
    return docs


def build_retriever():
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    docs = flatten_nss_json_for_documents()
    db = FAISS.from_documents(docs, embedding)
    return db.as_retriever(search_kwargs={"k": 4})
